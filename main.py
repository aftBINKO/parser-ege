"""Оркестратор пайплайна: парсинг → Gemini → публикация в админку.

Связывает четыре модуля в одну цепочку. На вход — номера задач с kompege, на
выходе — опубликованные разборы плюс отчёт о том, что не прошло.

Как устроена конкурентность::

    задача A:  [парсинг]──[  Gemini  ]──[публикация]
    задача B:            [парсинг]──[  Gemini  ]──[публикация]
    задача C:                      [парсинг]──[  Gemini  ]──...

Парсинг и публикация сериализованы семафорами: скрапер держит одну HTTP-сессию,
загрузчик — одну вкладку браузера, и параллелить их нельзя. Зато обращения к
Gemini идут пачкой, а они и есть самое медленное звено. В итоге пока одна задача
думает в модели, следующая уже скачивается.

Сверка ответов
--------------
Ответ модели сравнивается с ответом сайта (``site_answer``). При расхождении
задача **не публикуется**, а откладывается в ``mismatches.json`` на ручную
проверку: расхождение почти всегда означает, что модель ошиблась в решении, и
публиковать такой разбор вреднее, чем не публиковать ничего.

Порядок шагов здесь принципиален. Сверка идёт сразу после решения — до правок и
до переписывания условия, потому что ответ с kompege относится к **исходной**
формулировке. Стоит переставить переменные, и правильный ответ станет другим,
поэтому переписанный вариант проверяется не эталоном, а независимым повторным
решением (см. :func:`llm_processor.verify_rewrite`).

Запуск::

    python main.py 21401 21402 21403
    python main.py --numbers-file numbers.txt --dry-run
    python main.py 21401 --skip-upload                  # только парсинг и решение
    python main.py 21401 --review --style-file s.md     # с ручной правкой
    python main.py 21401 --rewrite                      # + уникализация условия
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Sequence

from dotenv import load_dotenv

from llm_processor import GeminiProcessor, LLMError, TaskSolution
from review import ReviewDecision, review_solution
from scraper import ScrapedTask, ScraperError, TaskScraper
from uploader import AdminUploader, AuthStateError, UploaderError, UploadResult

load_dotenv()

logger = logging.getLogger("pipeline")

#: Куда складывать готовые разборы.
DEFAULT_OUTPUT_DIR = Path("output")

#: Файл с задачами, отложенными из-за расхождения ответов.
DEFAULT_MISMATCH_FILE = Path("output/mismatches.json")


class ReviewAborted(Exception):
    """Пользователь прервал прогон на этапе просмотра."""


# --------------------------------------------------------------------------- #
# Результаты
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class TaskOutcome:
    """Что произошло с одной задачей на всём пути пайплайна.

    :param stage: на каком шаге всё закончилось — ``scrape``, ``llm``,
        ``compare``, ``upload`` или ``done``.
    """

    number: str
    stage: str
    ok: bool
    message: str = ""
    model_answer: str = ""
    site_answer: str = ""
    solution_path: Path | None = None

    def to_dict(self) -> dict[str, object]:
        """Словарь для отчёта."""
        return {
            "number": self.number,
            "stage": self.stage,
            "ok": self.ok,
            "message": self.message,
            "model_answer": self.model_answer,
            "site_answer": self.site_answer,
            "solution_path": str(self.solution_path) if self.solution_path else None,
        }


@dataclass(slots=True)
class PipelineReport:
    """Итог прогона по всем задачам."""

    outcomes: list[TaskOutcome] = field(default_factory=list)

    @property
    def published(self) -> list[TaskOutcome]:
        """Задачи, дошедшие до админки."""
        return [item for item in self.outcomes if item.ok and item.stage == "done"]

    @property
    def mismatched(self) -> list[TaskOutcome]:
        """Задачи, отложенные из-за расхождения ответов."""
        return [item for item in self.outcomes if item.stage == "compare"]

    @property
    def rejected(self) -> list[TaskOutcome]:
        """Задачи, отклонённые вручную при просмотре."""
        return [item for item in self.outcomes if item.stage == "review"]

    @property
    def failed(self) -> list[TaskOutcome]:
        """Задачи, упавшие с ошибкой."""
        return [
            item
            for item in self.outcomes
            if not item.ok and item.stage in {"scrape", "llm", "upload"}
        ]

    def summary(self) -> str:
        """Короткая сводка для лога."""
        parts = [
            f"всего {len(self.outcomes)}",
            f"опубликовано {len(self.published)}",
            f"расхождений {len(self.mismatched)}",
            f"ошибок {len(self.failed)}",
        ]
        if self.rejected:
            parts.insert(-1, f"отклонено вручную {len(self.rejected)}")
        return ", ".join(parts)


# --------------------------------------------------------------------------- #
# Сверка ответов
# --------------------------------------------------------------------------- #


def normalize_answer(answer: str) -> str:
    """Привести ответ к виду, пригодному для сравнения.

    Ответы ЕГЭ — короткие строки вроде ``xwzy``, ``42`` или ``12 34``. Различия в
    регистре и пробелах содержательными не являются, а вот порядок символов —
    ещё как: ``xwzy`` и ``xwyz`` это разные ответы, поэтому сортировать нельзя.
    """
    cleaned = re.sub(r"\s+", "", answer.strip().casefold())
    return cleaned.replace(",", "").replace(";", "").rstrip(".")


def answers_match(model_answer: str, site_answer: str) -> bool:
    """Совпадают ли ответы модели и сайта.

    Если сайт ответа не дал, сверять не с чем — считаем, что расхождения нет:
    иначе пайплайн замолчит на всех задачах без эталона.
    """
    if not site_answer:
        return True
    return normalize_answer(model_answer) == normalize_answer(site_answer)


# --------------------------------------------------------------------------- #
# Пайплайн
# --------------------------------------------------------------------------- #


class Pipeline:
    """Проводит задачи через парсинг, модель и публикацию.

    :param scraper: источник задач.
    :param processor: клиент Gemini.
    :param uploader: загрузчик в админку; ``None`` — публикация пропускается.
    :param output_dir: куда сохранять готовые разборы.
    :param llm_concurrency: сколько задач держать в модели одновременно.
    :param review: показывать каждый разбор на правку перед публикацией.
    :param rewrite: переписывать условие своими словами (антиплагиат).
    :param style_reference: образец оформления разбора для модели.
    """

    def __init__(
        self,
        scraper: TaskScraper,
        processor: GeminiProcessor,
        uploader: AdminUploader | None,
        *,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        llm_concurrency: int = 3,
        review: bool = False,
        rewrite: bool = False,
        style_reference: str = "",
    ) -> None:
        self.scraper = scraper
        self.processor = processor
        self.uploader = uploader
        self.output_dir = Path(output_dir)
        self.review = review
        self.rewrite = rewrite
        self.style_reference = style_reference

        # Скрапер и загрузчик — однопоточные ресурсы: одна HTTP-сессия и одна
        # вкладка браузера соответственно. Модель — единственное место, где
        # параллелизм реально помогает.
        self._scrape_lock = asyncio.Semaphore(1)
        self._upload_lock = asyncio.Semaphore(1)
        self._llm_limit = asyncio.Semaphore(max(1, llm_concurrency))

    # -- шаги ---------------------------------------------------------------- #

    async def _scrape(self, number: str) -> ScrapedTask:
        """Забрать задачу с сайта."""
        async with self._scrape_lock:
            return await asyncio.to_thread(self.scraper.scrape, number)

    async def _solve(self, task: ScrapedTask) -> TaskSolution:
        """Прогнать задачу через модель.

        В промпт уходит не только условие, но и содержимое прикреплённых файлов —
        см. :meth:`ScrapedTask.build_prompt_text`.
        """
        async with self._llm_limit:
            return await self.processor.process(
                task.build_prompt_text(),
                task_id=task.task_id,
                style_reference=self.style_reference,
            )

    async def _publish(self, solution: TaskSolution) -> UploadResult:
        """Опубликовать разбор в админке."""
        async with self._upload_lock:
            return await asyncio.to_thread(self.uploader.publish, solution)

    def _save_solution(self, solution: TaskSolution) -> Path:
        """Сохранить разбор на диск.

        Файл пишется всегда — и при удачной публикации, и при расхождении: это
        единственный способ не потерять работу модели, за которую уже заплачено.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"task_{solution.task_id or 'unknown'}.json"
        path.write_text(solution.to_json(), encoding="utf-8")
        return path

    # -- обработка одной задачи ----------------------------------------------- #

    async def handle(self, number: str) -> TaskOutcome:
        """Провести одну задачу через весь пайплайн.

        Исключения наружу не выпускаются: сбой одной задачи не должен ронять
        прогон. Единственное исключение — протухшая сессия админки, её
        пробрасываем, потому что дальше сыпаться будут все.

        :raises AuthStateError: сессия админки истекла.
        """
        try:
            task = await self._scrape(number)
        except ScraperError as exc:
            logger.error("Задача %s: парсинг не удался — %s", number, exc)
            return TaskOutcome(number, "scrape", False, str(exc))

        try:
            solution = await self._solve(task)
        except LLMError as exc:
            logger.error("Задача %s: модель не справилась — %s", number, exc)
            return TaskOutcome(number, "llm", False, str(exc))

        path = self._save_solution(solution)

        # Сверка идёт ДО правок: эталон с сайта относится к исходной
        # формулировке, а после переписывания условия он уже не применим.
        if not answers_match(solution.answer, task.site_answer):
            logger.warning(
                "Задача %s: расхождение ответов (модель «%s», сайт «%s») — не публикую",
                number,
                solution.answer,
                task.site_answer,
            )
            return TaskOutcome(
                number,
                "compare",
                False,
                "ответ модели не совпал с ответом сайта",
                solution.answer,
                task.site_answer,
                path,
            )

        # Правка и уникализация — после сверки: дальше эталона уже нет.
        if self.review or self.rewrite:
            try:
                review_result = await review_solution(
                    self.processor,
                    solution,
                    raw_text=task.build_prompt_text(),
                    site_answer=task.site_answer,
                    style_reference=self.style_reference,
                    auto_rewrite=self.rewrite,
                    preview_dir=self.output_dir,
                )
            except (LLMError, OSError, ValueError) as exc:
                # Разбор уже сохранён на диск — сбой просмотра не повод его терять.
                logger.error("Задача %s: правка не удалась — %s", number, exc)
                return TaskOutcome(
                    number, "review", False, f"сбой при правке: {exc}",
                    solution.answer, task.site_answer, path,
                )
            solution = review_result.solution
            path = self._save_solution(solution)

            if review_result.decision == ReviewDecision.QUIT:
                raise ReviewAborted(f"прогон остановлен на задаче {number}")
            if review_result.decision == ReviewDecision.REJECT:
                logger.info("Задача %s отклонена вручную — не публикую", number)
                return TaskOutcome(
                    number, "review", False, "отклонена при просмотре",
                    solution.answer, task.site_answer, path,
                )
            if review_result.rewritten and review_result.verified is False:
                logger.warning(
                    "Задача %s: переписанное условие не прошло контрольное решение "
                    "(«%s» против «%s») — откладываю",
                    number,
                    solution.answer,
                    review_result.control_answer,
                )
                return TaskOutcome(
                    number, "compare", False,
                    "контрольное решение переписанного условия не сошлось",
                    solution.answer, review_result.control_answer, path,
                )

        if self.uploader is None:
            return TaskOutcome(
                number, "done", True, "публикация пропущена",
                solution.answer, task.site_answer, path,
            )

        result = await self._publish(solution)
        return TaskOutcome(
            number,
            "done" if result.ok else "upload",
            result.ok,
            result.message,
            solution.answer,
            task.site_answer,
            path,
        )

    async def run(self, numbers: Sequence[str]) -> PipelineReport:
        """Провести пачку задач.

        :raises AuthStateError: сессия админки истекла — прогон останавливается.
        """
        report = PipelineReport()

        # С ручным просмотром параллелить нечего: человек всё равно смотрит
        # задачи по одной, а перемешанный вывод в терминале только мешает.
        if self.review or self.rewrite:
            for number in numbers:
                try:
                    report.outcomes.append(await self.handle(number))
                except ReviewAborted as exc:
                    logger.warning("%s", exc)
                    break
            return report

        tasks = [asyncio.create_task(self.handle(number)) for number in numbers]

        try:
            for coroutine in asyncio.as_completed(tasks):
                report.outcomes.append(await coroutine)
        except AuthStateError:
            for task in tasks:
                task.cancel()
            raise

        # as_completed возвращает в порядке готовности — восстанавливаем исходный
        order = {number: index for index, number in enumerate(numbers)}
        report.outcomes.sort(key=lambda item: order.get(item.number, 0))
        return report


# --------------------------------------------------------------------------- #
# Отчёты
# --------------------------------------------------------------------------- #


def save_mismatches(report: PipelineReport, path: Path) -> Path | None:
    """Сохранить задачи с расхождением ответов для ручной проверки.

    :returns: путь к файлу или ``None``, если расхождений не было.
    """
    if not report.mismatched:
        return None

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "tasks": [item.to_dict() for item in report.mismatched],
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path


def print_report(report: PipelineReport) -> None:
    """Вывести человекочитаемый итог прогона."""
    print("\n" + "=" * 70)
    print(f"ИТОГ: {report.summary()}")
    print("=" * 70)

    for item in report.outcomes:
        if item.ok:
            mark, note = "✓", item.message or "опубликовано"
        elif item.stage == "compare":
            mark, note = "≠", f"модель «{item.model_answer}», сайт «{item.site_answer}»"
        elif item.stage == "review":
            mark, note = "—", "отклонена при просмотре"
        else:
            mark, note = "✗", f"[{item.stage}] {item.message}"
        print(f"  {mark} {item.number:>8}  {note}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def read_numbers(args: argparse.Namespace) -> list[str]:
    """Собрать список номеров задач из аргументов или файла.

    В файле номера разделяются переносами строк, пробелами или запятыми;
    строки, начинающиеся с ``#``, считаются комментариями.
    """
    numbers: list[str] = [str(number).strip() for number in args.numbers]

    if args.numbers_file:
        content = args.numbers_file.read_text(encoding="utf-8")
        for line in content.splitlines():
            line = line.split("#", 1)[0]
            numbers.extend(part for part in re.split(r"[\s,;]+", line) if part)

    seen: set[str] = set()
    unique: list[str] = []
    for number in numbers:
        if number and number not in seen:
            seen.add(number)
            unique.append(number)
    return unique


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Разобрать аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description="Пайплайн: парсинг задач kompege → Gemini → админка."
    )
    parser.add_argument("numbers", nargs="*", help="Номера задач (например, 21401)")
    parser.add_argument(
        "--numbers-file",
        type=Path,
        default=None,
        help="Файл со списком номеров (по одному в строке)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Куда сохранять разборы (по умолчанию {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--mismatch-file",
        type=Path,
        default=DEFAULT_MISMATCH_FILE,
        help=f"Файл для расхождений (по умолчанию {DEFAULT_MISMATCH_FILE})",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="Сколько задач держать в модели одновременно (по умолчанию 3)",
    )
    parser.add_argument(
        "--review",
        action="store_true",
        help="Показывать каждый разбор на правку перед публикацией "
        "(правки формулируются репликами модели)",
    )
    parser.add_argument(
        "--rewrite",
        action="store_true",
        help="Переписывать условие своими словами и проверять результат "
        "независимым решением; включает --review",
    )
    parser.add_argument(
        "--style-file",
        type=Path,
        default=None,
        help="Образец оформления разбора: модель получит его сразу, "
        "и первый же вариант выйдет в нужном стиле",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Только распарсить и решить, в админку не ходить",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Заполнять форму админки, но не сохранять",
    )
    parser.add_argument(
        "--headed", action="store_true", help="Показать окно браузера (отладка)"
    )
    parser.add_argument(
        "--source",
        choices=("api", "browser"),
        default="api",
        help="Откуда брать задачи (по умолчанию api)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Подробный лог"
    )
    return parser.parse_args(argv)


async def run_pipeline(args: argparse.Namespace, numbers: list[str]) -> PipelineReport:
    """Собрать компоненты и провести прогон."""
    scraper = TaskScraper(source=args.source)
    processor = GeminiProcessor()
    uploader = (
        None
        if args.skip_upload
        else AdminUploader(
            dry_run=args.dry_run,
            headless=not args.headed,
            screenshot_dir=args.output_dir / "screenshots",
        )
    )

    style_reference = ""
    if args.style_file:
        style_reference = args.style_file.read_text(encoding="utf-8")

    pipeline = Pipeline(
        scraper,
        processor,
        uploader,
        output_dir=args.output_dir,
        llm_concurrency=args.concurrency,
        review=args.review or args.rewrite,
        rewrite=args.rewrite,
        style_reference=style_reference,
    )

    try:
        return await pipeline.run(numbers)
    finally:
        scraper.close()
        if uploader is not None:
            uploader.close()


def main(argv: Sequence[str] | None = None) -> int:
    """Точка входа CLI. Возвращает код возврата процесса."""
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not args.verbose:
        # Каждый запрос к Gemini печатает строку httpx и служебное сообщение
        # SDK — на прогоне из сотни задач полезный лог в этом тонет.
        for noisy in ("httpx", "google_genai.models", "google_genai.types"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    numbers = read_numbers(args)
    if not numbers:
        logger.error("Не задано ни одного номера задачи — нечего обрабатывать")
        return 1

    if (args.review or args.rewrite) and not sys.stdin.isatty():
        logger.error(
            "Режим правки требует интерактивного терминала — уберите --review/--rewrite"
        )
        return 1

    if args.style_file and not args.style_file.exists():
        logger.error("Файл образца %s не найден", args.style_file)
        return 1

    logger.info("К обработке %s задач(и): %s", len(numbers), ", ".join(numbers))

    try:
        report = asyncio.run(run_pipeline(args, numbers))
    except AuthStateError as exc:
        logger.error("%s", exc)
        return 2
    except (LLMError, UploaderError) as exc:
        logger.error("Пайплайн не запустился: %s", exc)
        return 1
    except KeyboardInterrupt:
        logger.warning("Прервано пользователем")
        return 130

    print_report(report)

    mismatch_path = save_mismatches(report, args.mismatch_file)
    if mismatch_path:
        logger.warning(
            "Расхождений %s — отложены в %s на ручную проверку",
            len(report.mismatched),
            mismatch_path,
        )

    return 0 if not report.failed else 1


if __name__ == "__main__":
    sys.exit(main())
