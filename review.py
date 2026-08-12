"""Просмотр и правка разбора перед публикацией — в диалоге с Gemini.

Модель пишет решение по-своему, и её текст может расходиться с тем, как объясняет
преподаватель. Поэтому между «решили» и «опубликовали» встаёт ручной шаг: разбор
показывается целиком, а правки формулируются репликами модели, а не руками —
«сделай через перебор», «добавь второй способ», «вот образец, приведи к такому
виду».

Здесь же выполняется переписывание условия своими словами: дословная копия чужого
текста на сайте школы нежелательна. Переписанный вариант **обязательно**
проверяется независимым решением — после правки условия ответ с kompege эталоном
быть перестаёт (перестановка переменных меняет и правильный ответ).

Команды диалога:

===========  =================================================================
``п``        показать разбор целиком
``р``        правка: своя реплика модели
``о``        правка по образцу из файла
``у``        уникализировать условие (переписать своими словами)
``н``        назад — откатить последнюю правку
``д``        принять разбор
``х``        отклонить задачу (не публиковать)
``в``        выйти из прогона
===========  =================================================================

Модуль работает и отдельно, над сохранённым разбором::

    python review.py output/task_21401.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import webbrowser
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Sequence

from dotenv import load_dotenv

from formatting import render_hints, render_html, render_plain
from llm_processor import (
    GeminiProcessor,
    LLMError,
    SolutionSession,
    TaskSolution,
    verify_rewrite,
)

load_dotenv()

logger = logging.getLogger(__name__)

#: Ширина разделителей при выводе.
WIDTH = 78


class ReviewDecision:
    """Чем закончился просмотр одной задачи."""

    ACCEPT = "accept"
    REJECT = "reject"
    QUIT = "quit"


@dataclass(slots=True)
class ReviewResult:
    """Итог просмотра: решение пользователя и актуальная версия разбора."""

    decision: str
    solution: TaskSolution
    rewritten: bool = False
    verified: bool | None = None
    control_answer: str = ""


# --------------------------------------------------------------------------- #
# Вывод
# --------------------------------------------------------------------------- #


def format_solution(
    solution: TaskSolution, *, site_answer: str = "", raw: bool = False
) -> str:
    """Собрать читаемое представление разбора для терминала.

    :param raw: показать исходный Markdown с LaTeX, как его вернула модель.
        По умолчанию текст показывается уже очищенным — ровно в том виде, в
        каком он попадёт в админку и достанется ученику.
    """
    prepare = (lambda text: text) if raw else render_plain
    hints = (
        solution.hints
        if raw
        else [line for line in render_hints(solution.hints).splitlines() if line]
    )

    lines = [
        "=" * WIDTH,
        f"ЗАДАЧА {solution.task_id}" + ("   [исходник модели]" if raw else ""),
        "=" * WIDTH,
        "",
        "УСЛОВИЕ:",
        prepare(solution.condition),
        "",
        "-" * WIDTH,
        "РЕШЕНИЕ:",
        prepare(solution.solution_text),
        "",
        "-" * WIDTH,
        f"ОТВЕТ: {solution.answer}",
    ]
    if site_answer:
        mark = "совпадает" if _same(solution.answer, site_answer) else "РАСХОЖДЕНИЕ"
        lines.append(f"Ответ сайта: {site_answer}  [{mark}]")

    if hints:
        lines.append("")
        lines.append("ПОДСКАЗКИ:")
        lines.extend(f"  {index}. {hint}" for index, hint in enumerate(hints, 1))
    lines.append("=" * WIDTH)
    return "\n".join(lines)


def build_preview(solution: TaskSolution, *, site_answer: str = "") -> str:
    """Собрать HTML-страницу предпросмотра — вид разбора глазами ученика."""
    answer_note = (
        f'<p class="meta">Ответ сайта: {escape(site_answer)}</p>' if site_answer else ""
    )
    hints = "".join(
        f"<li>{render_html(hint)}</li>"
        for hint in solution.hints
        if hint.strip()
    )
    return f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<title>Задача {escape(solution.task_id)} — предпросмотр</title>
<style>
  body {{ font: 16px/1.6 -apple-system, Segoe UI, Roboto, sans-serif;
         max-width: 820px; margin: 32px auto; padding: 0 20px; color: #1a1a1a; }}
  h1 {{ font-size: 20px; color: #666; font-weight: 500; }}
  h2 {{ font-size: 17px; margin-top: 32px; padding-bottom: 6px;
        border-bottom: 2px solid #e5e5e5; }}
  pre {{ background: #f6f8fa; padding: 12px 14px; border-radius: 6px;
         overflow-x: auto; font-size: 14px; }}
  code {{ font-family: ui-monospace, Menlo, Consolas, monospace; }}
  table {{ border-collapse: collapse; margin: 12px 0; }}
  th, td {{ border: 1px solid #999; padding: 5px 14px; text-align: center;
            min-width: 34px; }}
  .answer {{ font-size: 18px; font-weight: 600; background: #eef7ee;
             padding: 10px 14px; border-radius: 6px; display: inline-block; }}
  .meta {{ color: #888; font-size: 14px; }}
</style></head><body>
<h1>Задача {escape(solution.task_id)} — так это увидит ученик</h1>
<h2>Условие</h2>
{render_html(solution.condition)}
<h2>Решение</h2>
{render_html(solution.solution_text)}
<h2>Ответ</h2>
<p><span class="answer">{escape(solution.answer)}</span></p>
{answer_note}
<h2>Подсказки</h2>
<ol>{hints or "<li>нет</li>"}</ol>
</body></html>"""


def open_preview(
    solution: TaskSolution, *, site_answer: str = "", directory: Path | None = None
) -> Path:
    """Записать предпросмотр в файл и открыть его в браузере.

    :returns: путь к странице предпросмотра.
    """
    target = Path(directory or Path("output"))
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"preview_{solution.task_id or 'task'}.html"
    path.write_text(build_preview(solution, site_answer=site_answer), encoding="utf-8")

    try:
        webbrowser.open(path.resolve().as_uri())
    except Exception as exc:  # pragma: no cover - зависит от окружения
        logger.debug("Браузер не открылся: %s", exc)
    return path


def _same(left: str, right: str) -> bool:
    """Сравнить ответы без учёта регистра и пробелов."""
    return "".join(left.split()).casefold() == "".join(right.split()).casefold()


MENU = """
  п — показать разбор целиком      р — правка своей репликой
  о — правка по образцу из файла   у — уникализировать условие
  б — предпросмотр в браузере      и — показать исходник модели
  н — откатить последнюю правку    д — принять
  х — отклонить задачу             в — выйти из прогона
"""


# --------------------------------------------------------------------------- #
# Диалог
# --------------------------------------------------------------------------- #


def _read_line(prompt: str) -> str:
    """Прочитать строку, не роняя прогон на битом вводе.

    Вставка многострочного текста в терминал может разрезать кириллический
    символ между чтениями — Python отвечает на это ``UnicodeDecodeError``.
    Ронять из-за этого весь прогон (вместе с уже оплаченной работой модели)
    нельзя, поэтому битую строку просто просим повторить.
    """
    while True:
        try:
            return input(prompt).strip()
        except UnicodeDecodeError:
            print("Ввод не удалось прочитать (обрезанный символ). Повторите строку.")


async def _ask(prompt: str) -> str:
    """Спросить пользователя, не блокируя событийный цикл."""
    return await asyncio.to_thread(_read_line, prompt)


def _read_block(prompt: str) -> str:
    """Прочитать многострочный ввод до пустой строки.

    Правки естественно формулируются в несколько строк («подсказки должны быть
    такие: 1. … 2. …»). Обычный ``input`` берёт только первую, а остальные
    достаются следующему вопросу и выглядят как непонятные команды.
    """
    print(f"{prompt}\n(пустая строка — закончить ввод)")
    lines: list[str] = []
    while True:
        try:
            line = _read_line("| ")
        except EOFError:  # ввод закончился — считаем это концом блока
            break
        if not line:
            break
        lines.append(line)
    return "\n".join(lines).strip()


async def _ask_block(prompt: str) -> str:
    """Спросить многострочный текст, не блокируя событийный цикл."""
    return await asyncio.to_thread(_read_block, prompt)


async def review_solution(
    processor: GeminiProcessor,
    solution: TaskSolution,
    *,
    raw_text: str = "",
    site_answer: str = "",
    style_reference: str = "",
    auto_rewrite: bool = False,
    preview_dir: Path | None = None,
) -> ReviewResult:
    """Показать разбор и дать его поправить до публикации.

    Разбор показывается уже очищенным от разметки — в том виде, в каком он
    попадёт в админку. Исходник модели доступен по отдельной команде.

    :param raw_text: исходный текст задачи — модель получит его как контекст
        диалога. Если пусто, берётся условие из разбора.
    :param site_answer: ответ с сайта для сверки на экране.
    :param auto_rewrite: сразу переписать условие своими словами, не дожидаясь
        команды.
    :param preview_dir: куда складывать страницы предпросмотра.
    :returns: решение пользователя и актуальная версия разбора.
    """
    session = processor.start_session(
        raw_text or solution.condition, solution, style_reference=style_reference
    )
    history: list[TaskSolution] = [solution]
    rewritten = False
    verified: bool | None = None
    control_answer = ""

    print(format_solution(session.solution, site_answer=site_answer))

    if auto_rewrite:
        rewritten, verified, control_answer = await _do_rewrite(processor, session, history)

    while True:
        print(MENU)
        try:
            command = (await _ask("Действие: ")).lower()
        except EOFError:
            logger.warning("Ввод закрыт — считаю задачу отклонённой")
            return ReviewResult(ReviewDecision.REJECT, session.solution, rewritten,
                                verified, control_answer)

        if command in {"п", "p"}:
            print(format_solution(session.solution, site_answer=site_answer))

        elif command in {"и", "i"}:
            print(format_solution(session.solution, site_answer=site_answer, raw=True))

        elif command in {"б", "b"}:
            path = open_preview(
                session.solution, site_answer=site_answer, directory=preview_dir
            )
            print(f"Предпросмотр открыт в браузере: {path}")

        elif command in {"р", "r"}:
            instruction = await _ask_block("Что поправить:")
            if instruction:
                await _apply(session, history, instruction)
                print(format_solution(session.solution, site_answer=site_answer))

        elif command in {"о", "o"}:
            path_text = await _ask("Файл с образцом: ")
            sample = _read_sample(Path(path_text)) if path_text else ""
            if sample:
                instruction = (
                    await _ask("Что сделать с образцом [Enter — привести к нему]: ")
                    or "Приведи разбор к виду образца: та же структура и стиль подачи."
                )
                await _apply(session, history, instruction, sample=sample)
                print(format_solution(session.solution, site_answer=site_answer))

        elif command in {"у", "u"}:
            extra = await _ask_block(
                "Пожелания к переписыванию (пусто — по умолчанию):"
            )
            rewritten, verified, control_answer = await _do_rewrite(
                processor, session, history, extra
            )
            print(format_solution(session.solution, site_answer=site_answer))

        elif command in {"н", "n"}:
            if len(history) > 1:
                history.pop()
                session.solution = history[-1]
                print("Откатил последнюю правку.")
                print(format_solution(session.solution, site_answer=site_answer))
            else:
                print("Откатывать нечего — это первая версия.")

        elif command in {"д", "d"}:
            return ReviewResult(ReviewDecision.ACCEPT, session.solution, rewritten,
                                verified, control_answer)

        elif command in {"х", "x"}:
            return ReviewResult(ReviewDecision.REJECT, session.solution, rewritten,
                                verified, control_answer)

        elif command in {"в", "v", "q"}:
            return ReviewResult(ReviewDecision.QUIT, session.solution, rewritten,
                                verified, control_answer)

        else:
            print("Не понял команду.")


async def _apply(
    session: SolutionSession,
    history: list[TaskSolution],
    instruction: str,
    *,
    sample: str = "",
) -> None:
    """Отправить правку модели, сохранив предыдущую версию для отката."""
    print("Правлю...")
    try:
        await session.refine(instruction, sample=sample)
    except LLMError as exc:
        print(f"Не вышло: {exc}")
        return
    history.append(session.solution)


async def _do_rewrite(
    processor: GeminiProcessor,
    session: SolutionSession,
    history: list[TaskSolution],
    extra: str = "",
) -> tuple[bool, bool | None, str]:
    """Переписать условие и проверить результат независимым решением.

    :returns: тройка ``(переписано, проверка_сошлась, ответ_проверки)``.
    """
    print("Переписываю условие...")
    try:
        await session.rewrite(extra)
    except LLMError as exc:
        print(f"Не вышло: {exc}")
        return False, None, ""

    history.append(session.solution)

    print("Проверяю переписанное независимым решением...")
    try:
        matched, control_answer = await verify_rewrite(processor, session.solution)
    except LLMError as exc:
        print(f"Проверить не удалось: {exc}")
        return True, None, ""

    if matched:
        print(f"Проверка сошлась: ответ «{session.solution.answer}».")
    else:
        print(
            f"ВНИМАНИЕ: контрольное решение дало «{control_answer}», "
            f"а в разборе «{session.solution.answer}». "
            "Переписанное условие стоит проверить руками."
        )
    return True, matched, control_answer


def _read_sample(path: Path) -> str:
    """Прочитать файл с образцом оформления."""
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"Не удалось прочитать {path}: {exc}")
        return ""


# --------------------------------------------------------------------------- #
# CLI — правка сохранённого разбора
# --------------------------------------------------------------------------- #


def load_solution_file(path: Path) -> TaskSolution:
    """Прочитать разбор из JSON-файла.

    :raises ValueError: файл нечитаем или неполон.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Не удалось прочитать {path}: {exc}") from exc

    hints = data.get("hints") or []
    if isinstance(hints, str):
        hints = [hints]

    return TaskSolution(
        task_id=str(data.get("task_id", "")),
        condition=str(data.get("condition", "")),
        solution_text=str(data.get("solution_text", "")),
        answer=str(data.get("answer", "")),
        hints=[str(hint) for hint in hints],
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Разобрать аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description="Просмотр и правка разбора в диалоге с Gemini."
    )
    parser.add_argument("solution", type=Path, help="JSON с разбором")
    parser.add_argument(
        "--style-file",
        type=Path,
        default=None,
        help="Образец оформления разбора, который модель получит сразу",
    )
    parser.add_argument(
        "--rewrite",
        action="store_true",
        help="Сразу переписать условие своими словами",
    )
    parser.add_argument(
        "--site-answer", default="", help="Ответ сайта для сверки на экране"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Точка входа CLI. Возвращает код возврата процесса."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    args = _parse_args(argv)

    if not sys.stdin.isatty():
        logger.error("Нужен интерактивный терминал: правка идёт в диалоге")
        return 1

    try:
        solution = load_solution_file(args.solution)
        processor = GeminiProcessor()
    except (ValueError, LLMError) as exc:
        logger.error("%s", exc)
        return 1

    style = _read_sample(args.style_file) if args.style_file else ""

    result = asyncio.run(
        review_solution(
            processor,
            solution,
            site_answer=args.site_answer,
            style_reference=style,
            auto_rewrite=args.rewrite,
        )
    )

    if result.decision == ReviewDecision.ACCEPT:
        args.solution.write_text(result.solution.to_json(), encoding="utf-8")
        logger.info("Разбор сохранён в %s", args.solution)
        return 0

    logger.info("Изменения не сохранены (%s)", result.decision)
    return 0


if __name__ == "__main__":
    sys.exit(main())
