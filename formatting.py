"""Приведение разбора к виду, в котором его увидит ученик.

Модель пишет в Markdown с LaTeX: ``$F = x \\land y$``, ``**жирный**``,
ограждения ```` ```python ````. В редакторе админки это не разметка, а обычный
текст — ученик увидит доллары, звёздочки и обратные кавычки буквально.

Поэтому перед публикацией текст прогоняется через этот модуль:

* формулы разворачиваются в юникод — ``x ∧ (z → w) ∧ ¬y`` вместо
  ``$x \\land (z \\to w) \\land \\neg y$``;
* ограждения кода снимаются, сам код остаётся как есть;
* ``**жирный**``, ``*курсив*`` и заголовки ``###`` теряют служебные символы;
* степени и индексы становятся надстрочными и подстрочными знаками
  (``x^2`` → ``x²``), где для них есть юникод.

Второй режим — :func:`render_html`: та же нормализация, но с сохранением
структуры (код в ``<pre>``, таблицы в ``<table>``). Он нужен предпросмотру,
чтобы перед публикацией увидеть разбор глазами ученика.

Markdown остаётся тем, что хранится в JSON: это исходник, с ним удобно работать
модели. Рендер — только на выходе.
"""

from __future__ import annotations

import html
import re

#: Команды LaTeX и их юникодные соответствия.
LATEX_SYMBOLS: dict[str, str] = {
    # логика
    r"\land": "∧", r"\wedge": "∧", r"\lor": "∨", r"\vee": "∨",
    r"\neg": "¬", r"\lnot": "¬", r"\oplus": "⊕",
    r"\to": "→", r"\rightarrow": "→", r"\Rightarrow": "⇒",
    r"\leftrightarrow": "↔", r"\Leftrightarrow": "⇔", r"\implies": "⇒",
    r"\equiv": "≡", r"\forall": "∀", r"\exists": "∃",
    # сравнения и арифметика
    r"\le": "≤", r"\leq": "≤", r"\ge": "≥", r"\geq": "≥",
    r"\ne": "≠", r"\neq": "≠", r"\approx": "≈", r"\sim": "∼",
    r"\times": "×", r"\cdot": "·", r"\div": "÷", r"\pm": "±", r"\mp": "∓",
    r"\sqrt": "√", r"\sum": "∑", r"\prod": "∏", r"\infty": "∞",
    # множества
    r"\in": "∈", r"\notin": "∉", r"\subset": "⊂", r"\subseteq": "⊆",
    r"\cup": "∪", r"\cap": "∩", r"\emptyset": "∅",
    # греческие буквы, которые реально встречаются в задачах
    r"\alpha": "α", r"\beta": "β", r"\gamma": "γ", r"\delta": "δ",
    r"\pi": "π", r"\sigma": "σ", r"\omega": "ω", r"\lambda": "λ", r"\mu": "μ",
    # прочее
    r"\ldots": "…", r"\dots": "…", r"\cdots": "…", r"\quad": " ", r"\qquad": "  ",
}

#: Надстрочные знаки для степеней.
SUPERSCRIPTS = str.maketrans(
    "0123456789+-=()aeioruvxn", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ᵃᵉⁱᵒʳᵘᵛˣⁿ"
)

#: Подстрочные знаки для индексов.
SUBSCRIPTS = str.maketrans("0123456789+-=()aeioruvx", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑᵢₒᵣᵤᵥₓ")

#: Разделители формул: ``$$...$$``, ``$...$``, ``\(...\)``, ``\[...\]``.
MATH_PATTERN = re.compile(
    r"\$\$(?P<display>.+?)\$\$"
    r"|\$(?P<inline>[^$\n]+?)\$"
    r"|\\\((?P<paren>.+?)\\\)"
    r"|\\\[(?P<bracket>.+?)\\\]",
    re.DOTALL,
)

#: Блок кода с необязательным указанием языка.
CODE_BLOCK_PATTERN = re.compile(r"```[ \t]*(?P<lang>[\w+-]*)\n(?P<code>.*?)```", re.DOTALL)

#: Табличное окружение LaTeX — модель иногда рисует им таблицы истинности.
ARRAY_PATTERN = re.compile(
    r"\\begin\{(array|matrix|tabular)\}(\{[^}]*\})?(?P<body>.*?)\\end\{\1\}", re.DOTALL
)


# --------------------------------------------------------------------------- #
# Формулы
# --------------------------------------------------------------------------- #


def _convert_scripts(text: str) -> str:
    """Превратить степени и индексы в надстрочные и подстрочные знаки.

    Переводится только то, для чего есть юникод: ``x^2`` → ``x²``. Остальное
    остаётся как есть — лучше видимая «крышка», чем потерянный показатель.
    """

    def replace(match: re.Match[str], table: dict[int, int], marker: str) -> str:
        body = match.group(1) or match.group(2) or ""
        converted = body.translate(table)
        # перевелись не все символы — оставляем исходную запись
        return converted if converted != body or not body else f"{marker}{body}"

    text = re.sub(r"\^\{([^{}]*)\}|\^(\w)", lambda m: replace(m, SUPERSCRIPTS, "^"), text)
    text = re.sub(r"_\{([^{}]*)\}|_(\w)", lambda m: replace(m, SUBSCRIPTS, "_"), text)
    return text


def _convert_array(match: re.Match[str]) -> str:
    """Развернуть табличное окружение LaTeX в строки с разделителями."""
    rows: list[str] = []
    for raw_row in match.group("body").split(r"\\"):
        row = raw_row.replace(r"\hline", "").strip()
        if not row:
            continue
        cells = [cell.strip() for cell in row.split("&")]
        rows.append(" | ".join(cells))
    return "\n".join(rows)


def latex_to_unicode(formula: str) -> str:
    """Превратить формулу LaTeX в читаемый текст без служебных символов."""
    text = ARRAY_PATTERN.sub(_convert_array, formula)

    text = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", text)
    text = re.sub(r"\\(left|right|displaystyle|text|mathrm|mathit|operatorname)\b", "", text)
    text = re.sub(r"\\begin\{[^}]*\}|\\end\{[^}]*\}", "", text)

    # Длинные команды заменяем раньше коротких: \leq не должен пострадать от \le.
    for command in sorted(LATEX_SYMBOLS, key=len, reverse=True):
        text = text.replace(command, LATEX_SYMBOLS[command])

    text = _convert_scripts(text)
    text = text.replace(r"\,", " ").replace(r"\;", " ").replace(r"\!", "")
    text = text.replace("\\\\", "\n").replace("{", "").replace("}", "")
    text = re.sub(r"[ \t]{2,}", " ", text)
    # \neg y превращается в «¬ y» — отрицание пишется слитно с операндом
    text = re.sub(r"([¬√∀∃])\s+", r"\1", text)
    return text.strip()


def _replace_math(text: str) -> str:
    """Заменить все формулы в тексте их юникодным видом."""

    def replace(match: re.Match[str]) -> str:
        body = next(
            (value for value in match.groupdict().values() if value is not None), ""
        )
        converted = latex_to_unicode(body)
        # Выключная формула (``$$...$$``) стоит отдельным блоком.
        if match.group("display") is not None or match.group("bracket") is not None:
            return f"\n{converted}\n"
        return converted

    return MATH_PATTERN.sub(replace, text)


# --------------------------------------------------------------------------- #
# Простой текст
# --------------------------------------------------------------------------- #


def render_plain(text: str) -> str:
    """Привести Markdown с LaTeX к чистому тексту для формы админки.

    Что уходит: ограждения кода, символы ``*``, ``_``, ``#``, доллары формул.
    Что остаётся: сам код, структура абзацев и таблицы (в виде строк с ``|``).
    """
    if not text:
        return ""

    # Код вынимаем первым, чтобы его содержимое не пострадало от разметки.
    blocks: list[str] = []

    def stash(match: re.Match[str]) -> str:
        blocks.append(match.group("code").rstrip())
        return f"\x00{len(blocks) - 1}\x00"

    text = CODE_BLOCK_PATTERN.sub(stash, text)

    text = _replace_math(text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\*\*\*(.+?)\*\*\*", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"(?<!\w)\*(?!\s)(.+?)(?<!\s)\*(?!\w)", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"(?<!\w)__(.+?)__(?!\w)", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*+]\s+", "— ", text, flags=re.MULTILINE)

    def unstash(match: re.Match[str]) -> str:
        return blocks[int(match.group(1))]

    text = re.sub(r"\x00(\d+)\x00", unstash, text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def render_hints(hints: list[str], *, separator: str = "\n") -> str:
    """Собрать подсказки в одну строку, очистив разметку.

    Нумерация, которую модель иногда дублирует в тексте подсказки, снимается:
    в форме подсказки и так идут списком, и «1. 1. …» выглядит небрежно.
    """
    cleaned = []
    for hint in hints:
        text = render_plain(hint)
        text = re.sub(r"^\s*\d+[.)]\s*", "", text)
        if text:
            cleaned.append(text)
    return separator.join(cleaned)


# --------------------------------------------------------------------------- #
# HTML для предпросмотра
# --------------------------------------------------------------------------- #


def _table_to_html(lines: list[str]) -> str:
    """Собрать HTML-таблицу из строк Markdown-таблицы."""
    rows: list[list[str]] = []
    for line in lines:
        if re.fullmatch(r"\s*\|[\s|:-]+\|\s*", line):  # строка-разделитель
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        rows.append(cells)

    if not rows:
        return ""

    parts = ["<table>"]
    header, *body = rows
    parts.append(
        "<tr>" + "".join(f"<th>{html.escape(cell)}</th>" for cell in header) + "</tr>"
    )
    for row in body:
        parts.append(
            "<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>"
        )
    parts.append("</table>")
    return "".join(parts)


def render_html(text: str) -> str:
    """Превратить Markdown с LaTeX в HTML — так, как это увидит ученик.

    Используется предпросмотром: код остаётся моноширинным блоком, таблицы —
    таблицами, формулы — юникодом.
    """
    if not text:
        return ""

    blocks: list[str] = []

    def stash(match: re.Match[str]) -> str:
        blocks.append(match.group("code").rstrip())
        return f"\x00{len(blocks) - 1}\x00"

    text = CODE_BLOCK_PATTERN.sub(stash, text)
    text = _replace_math(text)

    parts: list[str] = []
    table_buffer: list[str] = []

    def flush_table() -> None:
        if table_buffer:
            parts.append(_table_to_html(table_buffer))
            table_buffer.clear()

    for paragraph in re.split(r"\n{2,}", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        lines = paragraph.splitlines()
        if all(line.strip().startswith("|") for line in lines):
            table_buffer.extend(lines)
            flush_table()
            continue

        flush_table()
        placeholder = re.fullmatch(r"\x00(\d+)\x00", paragraph)
        if placeholder:
            code = html.escape(blocks[int(placeholder.group(1))])
            parts.append(f"<pre><code>{code}</code></pre>")
            continue

        escaped = html.escape(paragraph)
        escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped, flags=re.DOTALL)
        escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
        escaped = re.sub(r"^#{1,6}\s*(.+)$", r"<strong>\1</strong>", escaped, flags=re.MULTILINE)
        escaped = re.sub(r"\x00(\d+)\x00", lambda m: html.escape(blocks[int(m.group(1))]), escaped)
        parts.append("<p>" + escaped.replace("\n", "<br>") + "</p>")

    flush_table()
    return "\n".join(parts)
