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
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from dotenv import load_dotenv

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


def format_solution(solution: TaskSolution, *, site_answer: str = "") -> str:
    """Собрать читаемое представление разбора для терминала."""
    lines = [
        "=" * WIDTH,
        f"ЗАДАЧА {solution.task_id}",
        "=" * WIDTH,
        "",
        "УСЛОВИЕ:",
        solution.condition,
        "",
        "-" * WIDTH,
        "РЕШЕНИЕ:",
        solution.solution_text,
        "",
        "-" * WIDTH,
        f"ОТВЕТ: {solution.answer}",
    ]
    if site_answer:
        mark = "совпадает" if _same(solution.answer, site_answer) else "РАСХОЖДЕНИЕ"
        lines.append(f"Ответ сайта: {site_answer}  [{mark}]")

    if solution.hints:
        lines.append("")
        lines.append("ПОДСКАЗКИ:")
        lines.extend(f"  {index}. {hint}" for index, hint in enumerate(solution.hints, 1))
    lines.append("=" * WIDTH)
    return "\n".join(lines)


def _same(left: str, right: str) -> bool:
    """Сравнить ответы без учёта регистра и пробелов."""
    return "".join(left.split()).casefold() == "".join(right.split()).casefold()


MENU = """
  п — показать разбор целиком      р — правка своей репликой
  о — правка по образцу из файла   у — уникализировать условие
  н — откатить последнюю правку    д — принять
  х — отклонить задачу             в — выйти из прогона
"""


# --------------------------------------------------------------------------- #
# Диалог
# --------------------------------------------------------------------------- #


async def _ask(prompt: str) -> str:
    """Спросить пользователя, не блокируя событийный цикл."""
    return (await asyncio.to_thread(input, prompt)).strip()


async def review_solution(
    processor: GeminiProcessor,
    solution: TaskSolution,
    *,
    raw_text: str = "",
    site_answer: str = "",
    style_reference: str = "",
    auto_rewrite: bool = False,
) -> ReviewResult:
    """Показать разбор и дать его поправить до публикации.

    :param raw_text: исходный текст задачи — модель получит его как контекст
        диалога. Если пусто, берётся условие из разбора.
    :param site_answer: ответ с сайта для сверки на экране.
    :param auto_rewrite: сразу переписать условие своими словами, не дожидаясь
        команды.
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

        elif command in {"р", "r"}:
            instruction = await _ask("Что поправить: ")
            if instruction:
                await _apply(session, history, instruction)
                print(format_solution(session.solution, site_answer=site_answer))

        elif command in {"о", "o"}:
            path = await _ask("Файл с образцом: ")
            sample = _read_sample(Path(path)) if path else ""
            if sample:
                instruction = (
                    await _ask("Что сделать с образцом [Enter — привести к нему]: ")
                    or "Приведи разбор к виду образца: та же структура и стиль подачи."
                )
                await _apply(session, history, instruction, sample=sample)
                print(format_solution(session.solution, site_answer=site_answer))

        elif command in {"у", "u"}:
            extra = await _ask("Пожелания к переписыванию [Enter — по умолчанию]: ")
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
