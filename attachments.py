"""Работа с вложениями задач: картинки, таблицы и документы.

К задачам ЕГЭ прикладывают не только картинки: задания 9 и 24–27 идут с файлами
данных — таблицами (``.ods``, ``.csv``) и документами (``.odt``, ``.txt``).
Модель не увидит содержимое файла по ссылке, поэтому текст из вложений
извлекается локально и подмешивается в промпт.

ODS и ODT — это zip-архивы с XML внутри, поэтому разбор сделан на стандартной
библиотеке (``zipfile`` + ``xml.etree``): лишняя зависимость ради двух форматов
не нужна. Таблицы (и отдельные файлы, и таблицы внутри документов) приводятся к
Markdown — тому же виду, в котором в промпт попадают HTML-таблицы условия, так
что модель видит единый формат.

Содержимое обрезается по :data:`DEFAULT_MAX_CHARS`: в задании 24 строка может
весить мегабайты, а платить за неё токенами смысла нет — для разбора хватает
начала, и факт обрезки честно помечается в тексте.
"""

from __future__ import annotations

import csv
import io
import logging
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree

logger = logging.getLogger(__name__)

#: Сколько символов содержимого вложения оставлять для промпта.
DEFAULT_MAX_CHARS = 20_000

#: Кодировки, которыми пробуем читать текстовые файлы. cp1251 обязателен:
#: файлы к задачам ЕГЭ часто сохранены именно в ней.
TEXT_ENCODINGS = ("utf-8-sig", "utf-8", "cp1251", "koi8-r")

#: Тип вложения по расширению файла.
KIND_BY_EXTENSION: dict[str, str] = {
    ".png": "image", ".jpg": "image", ".jpeg": "image", ".gif": "image",
    ".webp": "image", ".bmp": "image", ".svg": "image",
    ".ods": "table", ".csv": "table", ".tsv": "table",
    ".odt": "document", ".rtf": "document",
    ".txt": "text", ".dat": "text", ".log": "text",
}

#: Пространства имён OpenDocument.
ODF_NS = {
    "office": "urn:oasis:names:tc:opendocument:xmlns:office:1.0",
    "table": "urn:oasis:names:tc:opendocument:xmlns:table:1.0",
    "text": "urn:oasis:names:tc:opendocument:xmlns:text:1.0",
}

#: Предохранитель от «широких» ODS: пустые ячейки там повторяются тысячами.
MAX_REPEAT = 100


class AttachmentError(Exception):
    """Не удалось прочитать содержимое вложения."""


# --------------------------------------------------------------------------- #
# Модель данных
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Attachment:
    """Скачанное вложение задачи.

    :param url: исходная ссылка.
    :param path: путь к локальной копии.
    :param name: имя файла, как оно было на сайте.
    :param kind: ``image`` | ``table`` | ``document`` | ``text`` | ``other``.
    :param alt: подпись (для картинок из HTML условия).
    :param text: извлечённое содержимое. Для картинок всегда пусто — их
        содержимое передаётся модели не текстом.
    :param truncated: содержимое было обрезано по лимиту.
    """

    url: str
    path: Path
    name: str = ""
    kind: str = "other"
    alt: str = ""
    text: str = ""
    truncated: bool = False

    @property
    def is_image(self) -> bool:
        """Является ли вложение картинкой."""
        return self.kind == "image"

    def to_dict(self) -> dict[str, Any]:
        """Словарь для сериализации (``Path`` приводится к строке)."""
        data = asdict(self)
        data["path"] = str(self.path)
        return data


# --------------------------------------------------------------------------- #
# Общее
# --------------------------------------------------------------------------- #


def detect_kind(name: str) -> str:
    """Определить тип вложения по имени файла."""
    return KIND_BY_EXTENSION.get(Path(name).suffix.lower(), "other")


def rows_to_markdown(rows: list[list[str]]) -> str:
    """Собрать Markdown-таблицу из строк.

    Пустые ячейки сохраняются: в задачах ЕГЭ пропуск часто и есть условие.
    Первая строка считается заголовком.
    """
    rows = [row for row in rows if row]
    if not rows:
        return ""

    width = max(len(row) for row in rows)
    normalized = [
        [cell.replace("|", "\\|").replace("\n", " ").strip() for cell in row]
        + [""] * (width - len(row))
        for row in rows
    ]

    lines = [
        "| " + " | ".join(normalized[0]) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in normalized[1:])
    return "\n".join(lines)


def _trim_trailing_empty(rows: list[list[str]]) -> list[list[str]]:
    """Убрать хвостовые пустые строки и столбцы.

    Табличные редакторы щедро сохраняют пустое пространство листа — без чистки
    в промпт уедут сотни пустых строк.
    """
    while rows and not any(cell.strip() for cell in rows[-1]):
        rows.pop()
    if not rows:
        return []

    width = max(len(row) for row in rows)
    while width > 0 and all(
        not (row[width - 1].strip() if width - 1 < len(row) else "") for row in rows
    ):
        width -= 1
    return [row[:width] for row in rows]


# --------------------------------------------------------------------------- #
# Чтение форматов
# --------------------------------------------------------------------------- #


def read_text_file(path: Path) -> str:
    """Прочитать текстовый файл, подобрав кодировку.

    :raises AttachmentError: ни одна из кодировок не подошла.
    """
    raw = path.read_bytes()
    for encoding in TEXT_ENCODINGS:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise AttachmentError(f"Не удалось определить кодировку файла {path.name}")


def read_csv_file(path: Path) -> str:
    """Прочитать CSV/TSV и вернуть его Markdown-таблицей.

    Разделитель определяется автоматически; если определить не вышло —
    содержимое возвращается как обычный текст.
    """
    content = read_text_file(path)
    sample = content[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except csv.Error:
        logger.debug("Не удалось определить разделитель в %s — отдаю текстом", path.name)
        return content

    rows = [list(row) for row in csv.reader(io.StringIO(content), dialect)]
    return rows_to_markdown(_trim_trailing_empty(rows)) or content


def _odf_cell_text(cell: ElementTree.Element) -> str:
    """Собрать текст ячейки или абзаца ODF, склеивая вложенные ``text:p``."""
    parts = [
        "".join(paragraph.itertext()).strip()
        for paragraph in cell.iter(f"{{{ODF_NS['text']}}}p")
    ]
    return "\n".join(part for part in parts if part)


def _odf_table_rows(table: ElementTree.Element) -> list[list[str]]:
    """Развернуть таблицу ODF в список строк с учётом повторов ячеек."""
    rows: list[list[str]] = []
    for row in table.iter(f"{{{ODF_NS['table']}}}table-row"):
        cells: list[str] = []
        for cell in row.findall(f"{{{ODF_NS['table']}}}table-cell"):
            repeat = int(cell.get(f"{{{ODF_NS['table']}}}number-columns-repeated", "1"))
            value = _odf_cell_text(cell)
            cells.extend([value] * min(repeat, MAX_REPEAT))
        row_repeat = int(row.get(f"{{{ODF_NS['table']}}}number-rows-repeated", "1"))
        rows.extend([list(cells)] * min(row_repeat, MAX_REPEAT))
    return _trim_trailing_empty(rows)


def _open_odf_content(path: Path) -> ElementTree.Element:
    """Достать и разобрать ``content.xml`` из ODF-архива.

    :raises AttachmentError: файл не ODF или повреждён.
    """
    try:
        with zipfile.ZipFile(path) as archive:
            raw = archive.read("content.xml")
    except (zipfile.BadZipFile, KeyError, OSError) as exc:
        raise AttachmentError(f"{path.name}: не похоже на ODF-файл ({exc})") from exc

    try:
        return ElementTree.fromstring(raw)
    except ElementTree.ParseError as exc:
        raise AttachmentError(f"{path.name}: битый content.xml ({exc})") from exc


def read_ods(path: Path) -> str:
    """Прочитать таблицу ODS и вернуть листы в виде Markdown-таблиц.

    Листы разделяются заголовком с их именем — в задачах с несколькими листами
    важно понимать, где заканчивается один и начинается другой.
    """
    root = _open_odf_content(path)
    blocks: list[str] = []

    for table in root.iter(f"{{{ODF_NS['table']}}}table"):
        name = table.get(f"{{{ODF_NS['table']}}}name", "")
        markdown = rows_to_markdown(_odf_table_rows(table))
        if not markdown:
            continue
        blocks.append(f"### Лист «{name}»\n{markdown}" if name else markdown)

    if not blocks:
        raise AttachmentError(f"{path.name}: в таблице нет данных")
    return "\n\n".join(blocks)


def read_odt(path: Path) -> str:
    """Прочитать документ ODT: абзацы, заголовки и таблицы по порядку."""
    root = _open_odf_content(path)
    body = root.find(f"{{{ODF_NS['office']}}}body")
    document = body.find(f"{{{ODF_NS['office']}}}text") if body is not None else None
    if document is None:
        raise AttachmentError(f"{path.name}: в документе нет текстовой части")

    text_ns, table_ns = ODF_NS["text"], ODF_NS["table"]
    blocks: list[str] = []

    for node in document:
        tag = node.tag
        if tag in {f"{{{text_ns}}}p", f"{{{text_ns}}}h"}:
            line = "".join(node.itertext()).strip()
            if line:
                blocks.append(line)
        elif tag == f"{{{table_ns}}}table":
            markdown = rows_to_markdown(_odf_table_rows(node))
            if markdown:
                blocks.append(markdown)
        elif tag == f"{{{text_ns}}}list":
            for item in node.iter(f"{{{text_ns}}}p"):
                line = "".join(item.itertext()).strip()
                if line:
                    blocks.append(f"- {line}")

    if not blocks:
        raise AttachmentError(f"{path.name}: документ пуст")
    return "\n\n".join(blocks)


# --------------------------------------------------------------------------- #
# Точка входа модуля
# --------------------------------------------------------------------------- #


def extract_text(path: Path, *, max_chars: int = DEFAULT_MAX_CHARS) -> tuple[str, bool]:
    """Извлечь текстовое содержимое вложения.

    Картинки пропускаются: их содержимое текстом не передаётся.

    :returns: пара ``(текст, обрезано_ли)``. Для неподдерживаемых форматов —
        пустая строка.
    :raises AttachmentError: файл заявленного типа, но прочитать его не вышло.
    """
    suffix = path.suffix.lower()
    kind = detect_kind(path.name)

    if kind == "image":
        return "", False

    if suffix == ".ods":
        content = read_ods(path)
    elif suffix == ".odt":
        content = read_odt(path)
    elif suffix in {".csv", ".tsv"}:
        content = read_csv_file(path)
    elif kind == "text":
        content = read_text_file(path)
    else:
        logger.info("Формат %s пока не разбирается — файл только скачан", suffix or "?")
        return "", False

    content = content.strip()
    if len(content) <= max_chars:
        return content, False

    marker = f"\n\n[…обрезано, всего {len(content)} символов]"
    return content[:max_chars].rstrip() + marker, True


def describe_attachments(
    items: Iterable[Attachment], *, include_empty: bool = False
) -> str:
    """Собрать блок с содержимым вложений для промпта.

    Каждое вложение идёт под своим именем — модель должна понимать, из какого
    файла взяты данные, когда файлов несколько.

    :param include_empty: включать ли вложения без извлечённого текста
        (картинки, неподдерживаемые форматы) — упоминанием, без содержимого.
    """
    blocks: list[str] = []
    for item in items:
        title = item.name or Path(item.path).name
        if item.text:
            blocks.append(f"--- Файл «{title}» ({item.kind}) ---\n{item.text}")
        elif include_empty:
            blocks.append(f"--- Файл «{title}» ({item.kind}): содержимое не текстовое ---")
    return "\n\n".join(blocks)
