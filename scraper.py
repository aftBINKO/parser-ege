"""Парсинг задачи с kompege.ru: условие, формулы, таблицы, изображения, ответ.

Основной источник данных — **JSON API** сайта: ``GET /api/v1/task/<номер>``.
Это тот же запрос, который делает фронтенд, поэтому браузер для получения
условия не нужен: хватает ``requests``. Формулы приходят исходным LaTeX (KaTeX
рендерит их уже на клиенте), что для LLM куда полезнее отрендеренных глифов.

Резервный источник — **браузер** (``source="browser"``): Playwright открывает
``/task``, вводит номер в форму «Поиск по номеру» и разбирает разметку. Нужен,
если API изменится или закроется.

Что модуль делает с содержимым:

* HTML условия превращается в текст, пригодный для промпта;
* таблицы (например, таблицы истинности) конвертируются в Markdown с
  сохранением пустых ячеек — в таких задачах пропуск является частью условия;
* формулы восстанавливаются в LaTeX, даже если разметка уже отрендерена
  движком (``annotation[encoding="application/x-tex"]``, ``script[type="math/tex"]``);
* вложения скачиваются локально, а содержимое таблиц и документов (``.ods``,
  ``.odt``, ``.txt``, ``.csv``) распаковывается модулем :mod:`attachments` и
  подмешивается в промпт — задания 9 и 24–27 без файлов данных не решаются;
* забирается официальный ответ сайта (тот, что прячется за «Показать ответ») —
  его можно сравнивать с ответом Gemini как страховку от галлюцинаций.

Схема ответа API заранее не известна, поэтому поля ищутся по списку вероятных
имён (:class:`ApiFieldConfig`). Посмотреть реальный JSON и зафиксировать точные
имена::

    python scraper.py 21401 --dump-json raw.json

Пример использования::

    with TaskScraper() as scraper:
        task = scraper.scrape(21401)
        print(task.raw_text, task.site_answer)
"""

from __future__ import annotations

import argparse
import json
import logging
import mimetypes
import os
import re
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, NavigableString, Tag
from dotenv import load_dotenv

from attachments import (
    DEFAULT_MAX_CHARS,
    Attachment,
    AttachmentError,
    describe_attachments,
    detect_kind,
    extract_text,
    rows_to_markdown,
)

load_dotenv()

logger = logging.getLogger(__name__)

#: База JSON API сайта.
DEFAULT_API_URL = os.getenv("KOMPEGE_API_URL", "https://kompege.ru/api/v1")

#: Страница поиска задачи — источник для резервного, браузерного режима.
DEFAULT_TASK_URL = os.getenv("KOMPEGE_TASK_URL", "https://kompege.ru/task")

#: Куда складывать скачанные вложения (папка в .gitignore).
DEFAULT_DOWNLOAD_DIR = Path("downloads")

#: База для относительных ссылок на файлы задач.
DEFAULT_FILES_BASE_URL = os.getenv("KOMPEGE_FILES_URL", "https://kompege.ru/")

#: User-Agent обычного браузера: часть сайтов режет дефолтный UA requests.
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)

#: Скрипты матдвижков, блокируемые в браузерном режиме ради исходного LaTeX.
MATH_SCRIPT_PATTERNS = ("**/*mathjax*", "**/*MathJax*", "**/*katex*", "**/*KaTeX*")

#: Предохранитель от гигантских файлов при скачивании вложений, байты.
MAX_FILE_BYTES = 20 * 1024 * 1024


class ScraperError(Exception):
    """Не удалось получить или разобрать задачу."""


# --------------------------------------------------------------------------- #
# Конфигурация
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class ApiFieldConfig:
    """Имена полей в ответе API.

    Первым в каждом кортеже стоит имя из реальной схемы ``/api/v1/task/<номер>``,
    дальше — запасные варианты на случай изменения API. Порядок значим: имя
    перебирается слева направо, поэтому точное всегда выигрывает у запасного.

    Отдельно стоит отметить два поля, где интуиция подводит:

    * номер задачи — это ``taskId`` (21401), тогда как ``id`` содержит
      внутренний UUID, а ``number`` — номер задания в ЕГЭ (1–27);
    * ответ лежит в ``key``, а не в ``answer``.

    Все имена сравниваются в нижнем регистре.
    """

    condition: tuple[str, ...] = ("text", "condition", "statement", "body", "html")
    answer: tuple[str, ...] = ("key", "answer", "correct_answer", "right_answer")
    task_id: tuple[str, ...] = ("taskid", "task_id", "id")
    images: tuple[str, ...] = ("files", "images", "pictures", "attachments", "media")
    solution: tuple[str, ...] = ("solve_text", "solution", "explanation")
    table: tuple[str, ...] = ("table", "tables", "grid")


@dataclass(slots=True)
class SelectorConfig:
    """Как найти элементы на странице — только для браузерного режима.

    Поиск формы намеренно идёт по видимому тексту, а не по CSS-классам: подпись
    «Номер задачи» стабильнее сгенерированных классов.

    :param number_input_placeholder: placeholder поля ввода номера.
    :param submit_button_text: текст кнопки отправки формы.
    :param show_answer_text: текст ссылки, раскрывающей ответ.
    :param container_css: CSS-селектор блока задачи. Если ``None`` — блок ищется
        эвристикой по заголовку вида «№ 21401».
    :param images: селектор изображений внутри блока задачи.
    :param drop_css: селекторы мусора, удаляемого из текста.
    :param drop_text: элементы, чей текст точно совпадает с одной из строк,
        удаляются целиком (кнопки-ссылки интерфейса).
    """

    number_input_placeholder: str = "Номер задачи"
    submit_button_text: str = "Показать задачу"
    show_answer_text: str = "Показать ответ"
    container_css: str | None = None
    images: str = "img"
    drop_css: tuple[str, ...] = ("script", "style", "noscript")
    drop_text: tuple[str, ...] = ("Показать ответ", "Показать решение")


# --------------------------------------------------------------------------- #
# Модель данных
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class ScrapedTask:
    """Результат парсинга одной задачи — вход для ``llm_processor.py``.

    :param site_answer: официальный ответ с сайта. Пустая строка, если сайт его
        не отдал. Не подставляется в ответ модели: это независимая величина для
        сверки.
    :param attachments: все вложения задачи — и картинки из условия, и файлы
        данных (``.txt``, ``.ods``, ``.odt``, ``.csv``) из поля ``files``.
    """

    task_id: str
    raw_text: str
    url: str = ""
    site_answer: str = ""
    attachments: list[Attachment] = field(default_factory=list)
    source: str = "api"

    @property
    def images(self) -> list[Attachment]:
        """Только картинки — их передают модели отдельно, а не текстом."""
        return [item for item in self.attachments if item.is_image]

    @property
    def data_files(self) -> list[Attachment]:
        """Только файлы данных: таблицы, документы, текстовые файлы."""
        return [item for item in self.attachments if not item.is_image]

    def build_prompt_text(self) -> str:
        """Собрать текст для LLM: условие плюс содержимое файлов данных.

        Модель не откроет файл по ссылке, поэтому распакованные таблицы и
        документы подмешиваются прямо в промпт — иначе задания 9 и 24–27
        решать не из чего.
        """
        block = describe_attachments(self.data_files)
        if not block:
            return self.raw_text
        return f"{self.raw_text}\n\nПрикреплённые файлы:\n\n{block}"

    def to_dict(self) -> dict[str, Any]:
        """Словарь, пригодный для сериализации в JSON."""
        data = asdict(self)
        data["attachments"] = [item.to_dict() for item in self.attachments]
        return data

    def to_json(self, *, indent: int = 2) -> str:
        """JSON-строка (UTF-8, без экранирования кириллицы)."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


# --------------------------------------------------------------------------- #
# Преобразование разметки в текст
# --------------------------------------------------------------------------- #


def _normalize_whitespace(text: str) -> str:
    """Схлопнуть пробелы и лишние переносы, убрать неразрывные пробелы."""
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _table_to_markdown(table: Tag) -> str:
    """Преобразовать HTML-таблицу в Markdown.

    Сборка Markdown общая с вложениями (:func:`attachments.rows_to_markdown`),
    чтобы таблицы из условия и таблицы из ODS выглядели в промпте одинаково.
    """
    rows: list[list[str]] = []
    for row in table.find_all("tr"):
        cells = row.find_all(["th", "td"])
        if cells:
            rows.append([_normalize_whitespace(cell.get_text(" ")) for cell in cells])
    return rows_to_markdown(rows)


def structured_table_to_markdown(table: Any) -> str:
    """Преобразовать табличные данные из JSON в Markdown.

    В ответе API есть поле ``table``; у задачи 21401 оно пустое, поэтому его
    форма достоверно не известна. Поддерживаются оба разумных варианта — список
    строк и объект с ключами вроде ``headers``/``rows``. Всё непонятное
    возвращается сырым JSON, чтобы данные не потерялись молча.
    """
    if not table:
        return ""

    if isinstance(table, dict):
        headers = table.get("headers") or table.get("columns") or []
        body = table.get("rows") or table.get("data") or []
        if body:
            rows = [[str(cell) for cell in row] for row in body if isinstance(row, list)]
            if headers:
                rows.insert(0, [str(cell) for cell in headers])
            markdown = rows_to_markdown(rows)
            if markdown:
                return markdown
        return json.dumps(table, ensure_ascii=False, indent=2)

    if isinstance(table, list):
        if all(isinstance(row, list) for row in table):
            return rows_to_markdown([[str(cell) for cell in row] for row in table])
        if all(isinstance(row, dict) for row in table) and table:
            headers = list(table[0].keys())
            rows = [headers] + [
                [str(row.get(key, "")) for key in headers] for row in table
            ]
            return rows_to_markdown(rows)

    return json.dumps(table, ensure_ascii=False, indent=2)


def _replace_math_nodes(soup: BeautifulSoup) -> None:
    """Вернуть формулам исходный LaTeX там, где движок уже отработал.

    В API-режиме формулы и так приходят исходником — это подстраховка для
    браузерного режима и для случая, когда сервер отдаёт готовую разметку:
    MathJax v2 хранит исходник в ``<script type="math/tex">``, а KaTeX и
    MathJax v3 — в ``annotation[encoding="application/x-tex"]``.
    """
    for script in soup.select('script[type^="math/tex"]'):
        latex = script.get_text().strip()
        script.replace_with(NavigableString(f" ${latex}$ " if latex else " "))

    for annotation in soup.select('annotation[encoding="application/x-tex"]'):
        latex = annotation.get_text().strip()
        # Заменяем весь контейнер формулы, иначе рядом останется отрендеренный вид.
        target: Tag = annotation
        for parent in annotation.parents:
            if not isinstance(parent, Tag):
                break
            classes = " ".join(parent.get("class", []))
            if parent.name == "mjx-container" or "katex" in classes:
                target = parent
                break
            if parent.name in {"math", "semantics"}:
                target = parent
        target.replace_with(NavigableString(f" ${latex}$ " if latex else " "))


def _looks_like_html(value: str) -> bool:
    """Грубая проверка, что строка — разметка, а не обычный текст."""
    return bool(re.search(r"<[a-zA-Z/][^>]*>", value))


# --------------------------------------------------------------------------- #
# Поиск полей в JSON неизвестной схемы
# --------------------------------------------------------------------------- #


def _walk(payload: Any) -> Iterable[dict[str, Any]]:
    """Обойти вложенные словари в ширину — от внешних к внутренним."""
    queue: deque[Any] = deque([payload])
    while queue:
        node = queue.popleft()
        if isinstance(node, dict):
            yield node
            queue.extend(node.values())
        elif isinstance(node, list):
            queue.extend(node)


def find_field(payload: Any, names: tuple[str, ...]) -> str:
    """Найти в JSON первое непустое скалярное значение по одному из имён.

    Имена перебираются в порядке кортежа: точное имя из реальной схемы должно
    выигрывать у запасного, даже если запасное встречается в JSON раньше. Внутри
    одного имени обход идёт в ширину, поэтому поле верхнего уровня выигрывает у
    одноимённого вложенного.

    :returns: значение строкой либо пустая строка, если ничего не нашлось.
    """
    for name in names:
        wanted = name.lower()
        for node in _walk(payload):
            for key, value in node.items():
                if key.lower() == wanted and isinstance(value, (str, int, float)):
                    text = str(value).strip()
                    if text:
                        return text
    return ""


def find_container(payload: Any, names: tuple[str, ...]) -> Any:
    """Найти в JSON первое непустое составное значение (словарь или список).

    Нужно для полей вроде ``table``, где данные лежат структурой, а не строкой.
    Имена перебираются в порядке кортежа — как и в :func:`find_field`.
    """
    for name in names:
        wanted = name.lower()
        for node in _walk(payload):
            for key, value in node.items():
                if key.lower() == wanted and isinstance(value, (dict, list)) and value:
                    return value
    return None


def find_attachment_refs(payload: Any, names: tuple[str, ...]) -> list[tuple[str, str]]:
    """Собрать ссылки на вложения из списковых полей JSON.

    Понимает список строк и список объектов (``{"url": ...}``,
    ``{"name": "24-1.txt"}``). Имя файла важно не меньше ссылки: по расширению
    определяется, как разбирать содержимое.

    :returns: список пар ``(ссылка_или_имя, имя_файла)``.
    """
    wanted = {name.lower() for name in names}
    refs: list[tuple[str, str]] = []

    for node in _walk(payload):
        for key, value in node.items():
            if key.lower() not in wanted or not isinstance(value, list):
                continue
            for item in value:
                if isinstance(item, str) and item.strip():
                    reference = item.strip()
                    refs.append((reference, Path(urlparse(reference).path).name))
                elif isinstance(item, dict):
                    reference = ""
                    for candidate in ("url", "src", "path", "link", "file", "filename", "name"):
                        found = item.get(candidate)
                        if isinstance(found, str) and found.strip():
                            reference = found.strip()
                            break
                    if not reference:
                        continue
                    name_value = item.get("name") or item.get("filename") or ""
                    name = (
                        str(name_value).strip()
                        or Path(urlparse(reference).path).name
                    )
                    refs.append((reference, name))
    return refs


# --------------------------------------------------------------------------- #
# Прочие утилиты
# --------------------------------------------------------------------------- #


def _pick_image_url(tag: Tag) -> str:
    """Выбрать ссылку на картинку с учётом ленивой загрузки и ``srcset``."""
    for attribute in ("src", "data-src", "data-original", "data-lazy-src"):
        value = tag.get(attribute)
        if isinstance(value, str) and value.strip():
            return value.strip()

    srcset = tag.get("srcset")
    if isinstance(srcset, str) and srcset.strip():
        # формат: "url1 1x, url2 2x" — берём первый вариант
        return srcset.split(",")[0].strip().split(" ")[0]
    return ""


def _guess_extension(url: str, name: str = "", content_type: str | None = None) -> str:
    """Подобрать расширение файла: по имени, затем по URL, затем по Content-Type.

    Расширение здесь не косметика: по нему :mod:`attachments` решает, разбирать
    файл как таблицу, документ или текст. Поэтому неизвестный тип честно
    остаётся без расширения, а не выдаётся за картинку.
    """
    for candidate in (Path(name).suffix, Path(urlparse(url).path).suffix):
        suffix = candidate.lower()
        if suffix and len(suffix) <= 6:
            return suffix

    if content_type:
        guessed = mimetypes.guess_extension(content_type.split(";")[0].strip())
        if guessed:
            return ".jpg" if guessed == ".jpe" else guessed
    return ""


#: JS-эвристика браузерного режима: найти блок задачи по заголовку «№ <номер>».
_CONTAINER_JS = """
([number, minLength]) => {
  const re = new RegExp('№\\\\s*' + number + '(\\\\D|$)');
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_ELEMENT);
  let deepest = null;
  while (walker.nextNode()) {
    const el = walker.currentNode;
    if (re.test(el.textContent || '')) deepest = el;
  }
  if (!deepest) return null;
  let current = deepest;
  for (let i = 0; i < 6 && current.parentElement; i++) {
    if ((current.textContent || '').length >= minLength) break;
    current = current.parentElement;
  }
  return current.outerHTML;
}
"""


# --------------------------------------------------------------------------- #
# Скрапер
# --------------------------------------------------------------------------- #


class TaskScraper:
    """Достаёт задачи по номеру: через JSON API либо через браузер.

    :param source: ``api`` (по умолчанию) или ``browser``.
    :param api_url: база API, например ``https://kompege.ru/api/v1``.
    :param task_url: страница формы — для браузерного режима и как Referer.
    :param fields: имена полей в ответе API.
    :param selectors: настройки браузерного режима.
    :param download_dir: куда складывать вложения.
    :param download_attachments: скачивать ли вложения.
    :param extract_attachment_text: распаковывать ли содержимое таблиц и
        документов для промпта.
    :param max_attachment_chars: сколько символов содержимого оставлять.
    :param files_base_url: база для относительных ссылок на файлы.
    :param fetch_answer: забирать ли официальный ответ сайта.
    :param headless: запускать браузер без окна (браузерный режим).
    :param block_math_js: блокировать матдвижок (браузерный режим).
    :param timeout: таймаут запросов и ожиданий, сек.
    :param request_delay: пауза между задачами, сек — вежливость к сайту.
    :param min_container_length: порог эвристики поиска блока задачи.
    """

    def __init__(
        self,
        *,
        source: str = "api",
        api_url: str = DEFAULT_API_URL,
        task_url: str = DEFAULT_TASK_URL,
        fields: ApiFieldConfig | None = None,
        selectors: SelectorConfig | None = None,
        download_dir: Path = DEFAULT_DOWNLOAD_DIR,
        download_attachments: bool = True,
        extract_attachment_text: bool = True,
        max_attachment_chars: int = DEFAULT_MAX_CHARS,
        files_base_url: str = DEFAULT_FILES_BASE_URL,
        fetch_answer: bool = True,
        headless: bool = True,
        block_math_js: bool = True,
        timeout: float = 30.0,
        request_delay: float = 1.0,
        min_container_length: int = 200,
    ) -> None:
        if source not in {"api", "browser"}:
            raise ValueError(f"Неизвестный источник: {source}")

        self.source = source
        self.api_url = api_url.rstrip("/")
        self.task_url = task_url
        self.fields = fields or ApiFieldConfig()
        self.selectors = selectors or SelectorConfig()
        self.download_dir = Path(download_dir)
        self.download_attachments = download_attachments
        self.extract_attachment_text = extract_attachment_text
        self.max_attachment_chars = max_attachment_chars
        self.files_base_url = files_base_url
        self.fetch_answer = fetch_answer
        self.headless = headless
        self.block_math_js = block_math_js
        self.timeout = timeout
        self.request_delay = request_delay
        self.min_container_length = min_container_length

        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": DEFAULT_USER_AGENT,
                "Accept": "application/json, text/plain, */*",
                # фронтенд ходит в API со страницы задачи — повторяем
                "Referer": self.task_url,
            }
        )
        self._last_request_at = 0.0

        self._playwright: Any = None
        self._browser: Any = None
        self._page: Any = None

    # -- общее --------------------------------------------------------------- #

    def _throttle(self) -> None:
        """Выдержать паузу между обращениями к сайту."""
        elapsed = time.monotonic() - self._last_request_at
        if self._last_request_at and elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self._last_request_at = time.monotonic()

    def close(self) -> None:
        """Закрыть браузер (если поднимался) и HTTP-сессию."""
        if self._browser is not None:
            try:
                self._browser.close()
            except Exception as exc:  # pragma: no cover - гасим шум при выходе
                logger.debug("Ошибка при закрытии браузера: %s", exc)
        if self._playwright is not None:
            try:
                self._playwright.stop()
            except Exception as exc:  # pragma: no cover
                logger.debug("Ошибка при остановке playwright: %s", exc)
        self._playwright = self._browser = self._page = None
        self.session.close()

    def __enter__(self) -> "TaskScraper":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- источник: JSON API --------------------------------------------------- #

    def fetch_payload(self, number: str) -> Any:
        """Забрать JSON задачи: ``GET {api_url}/task/{number}``.

        :raises ScraperError: сетевая ошибка, не-2xx или не-JSON в ответе.
        """
        url = f"{self.api_url}/task/{number}"
        self._throttle()
        logger.debug("GET %s", url)
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.JSONDecodeError as exc:
            raise ScraperError(f"{url} вернул не JSON: {exc}") from exc
        except requests.RequestException as exc:
            raise ScraperError(f"Не удалось получить {url}: {exc}") from exc

    def fetch_answer_payload(self, number: str) -> str:
        """Запросить ответ отдельным эндпоинтом, если его не было в основном JSON.

        Отсутствие такого эндпоинта — не ошибка: возвращается пустая строка.
        """
        url = f"{self.api_url}/task/{number}/answer"
        self._throttle()
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, requests.JSONDecodeError) as exc:
            logger.debug("Отдельный эндпоинт ответа недоступен (%s): %s", url, exc)
            return ""

        if isinstance(payload, (str, int, float)):
            return str(payload).strip()
        return find_field(payload, self.fields.answer)

    def parse_payload(self, payload: Any, number: str) -> ScrapedTask:
        """Собрать :class:`ScrapedTask` из ответа API.

        Условие может прийти как HTML или как обычный текст — распознаётся
        автоматически; HTML прогоняется через тот же разбор, что и в браузерном
        режиме (таблицы, формулы, картинки). Табличные данные из поля ``table``
        дописываются к условию отдельным блоком.

        :raises ScraperError: в ответе не нашлось текста условия.
        """
        condition = find_field(payload, self.fields.condition)
        if not condition:
            raise ScraperError(
                f"В ответе API нет текста условия задачи {number}. "
                "Посмотрите JSON через --dump-json и уточните ApiFieldConfig."
            )

        refs: list[tuple[str, str, str]] = []
        if _looks_like_html(condition):
            text, refs = self.parse_html(condition, self.task_url)
        else:
            text = _normalize_whitespace(condition)

        structured = structured_table_to_markdown(
            find_container(payload, self.fields.table)
        )
        if structured:
            text = f"{text}\n\n{structured}"

        known = {url for url, _, _ in refs}
        for reference, name in find_attachment_refs(payload, self.fields.images):
            absolute = urljoin(self.files_base_url, reference)
            if absolute not in known:
                known.add(absolute)
                refs.append((absolute, name, ""))

        answer = find_field(payload, self.fields.answer) if self.fetch_answer else ""
        if self.fetch_answer and not answer:
            answer = self.fetch_answer_payload(number)
        if answer and _looks_like_html(answer):
            answer, _ = self.parse_html(answer, self.task_url)

        task_id = find_field(payload, self.fields.task_id) or number

        return ScrapedTask(
            task_id=task_id,
            raw_text=text,
            url=f"{self.api_url}/task/{number}",
            site_answer=answer,
            attachments=self._collect_attachments(refs, number),
            source="api",
        )

    # -- источник: браузер ---------------------------------------------------- #

    def open_browser(self) -> None:
        """Поднять браузер и открыть страницу поиска задач.

        :raises ScraperError: playwright не установлен или страница не открылась.
        """
        if self._page is not None:
            return

        try:
            from playwright.sync_api import Error as PlaywrightError
            from playwright.sync_api import sync_playwright
        except ImportError as exc:  # pragma: no cover - зависит от окружения
            raise ScraperError(
                "Нужен playwright: pip install playwright && playwright install chromium"
            ) from exc

        self._playwright = sync_playwright().start()
        try:
            self._browser = self._playwright.chromium.launch(headless=self.headless)
            self._page = self._browser.new_page(user_agent=DEFAULT_USER_AGENT)
            self._page.set_default_timeout(int(self.timeout * 1000))

            if self.block_math_js:
                # Без матдвижка формулы остаются в DOM исходным LaTeX.
                for pattern in MATH_SCRIPT_PATTERNS:
                    self._page.route(pattern, lambda route: route.abort())

            logger.debug("Открываю %s", self.task_url)
            self._page.goto(self.task_url, wait_until="domcontentloaded")
        except PlaywrightError as exc:
            self.close()
            raise ScraperError(f"Не удалось открыть {self.task_url}: {exc}") from exc

    def _submit_number(self, number: str) -> None:
        """Ввести номер задачи в форму и дождаться появления задачи.

        :raises ScraperError: форма не найдена или задача не появилась.
        """
        from playwright.sync_api import Error as PlaywrightError

        page = self._page
        try:
            field_locator = page.get_by_placeholder(
                self.selectors.number_input_placeholder
            )
            field_locator.wait_for(state="visible")
            field_locator.fill(number)
            page.get_by_role(
                "button", name=self.selectors.submit_button_text
            ).first.click()
            page.wait_for_selector(rf"text=/№\s*{re.escape(number)}(\D|$)/")
        except PlaywrightError as exc:
            raise ScraperError(
                f"Не удалось открыть задачу {number}: {exc}. "
                "Проверьте SelectorConfig (подписи формы могли измениться)."
            ) from exc

    def _extract_container_html(self, number: str) -> str:
        """Достать HTML блока задачи: по селектору, иначе эвристикой.

        :raises ScraperError: блок не найден ни тем, ни другим способом.
        """
        page = self._page

        if self.selectors.container_css:
            node = page.query_selector(self.selectors.container_css)
            if node is not None:
                return node.inner_html()
            logger.warning(
                "Селектор '%s' не сработал — включаю эвристику по номеру задачи",
                self.selectors.container_css,
            )

        html = page.evaluate(_CONTAINER_JS, [number, self.min_container_length])
        if not html:
            raise ScraperError(
                f"Не нашёл блок задачи {number} на странице. "
                "Задайте container_css в SelectorConfig."
            )
        logger.debug("Блок задачи %s найден эвристикой", number)
        return html

    def _reveal_answer(self, number: str, text_before: str) -> str:
        """Кликнуть «Показать ответ» и вернуть появившийся текст.

        Ответ вычисляется как приращение текста блока: так не нужно угадывать
        селектор элемента, в котором сайт его показывает.

        :returns: текст ответа либо пустая строка, если ничего не изменилось.
        """
        from playwright.sync_api import Error as PlaywrightError

        page = self._page
        try:
            link = page.get_by_text(self.selectors.show_answer_text, exact=True).first
            if link.count() == 0:
                return ""
            link.click()
            # Ответ может подгружаться запросом — ждём изменения текста блока.
            page.wait_for_timeout(500)
        except PlaywrightError as exc:
            logger.debug("Не удалось раскрыть ответ задачи %s: %s", number, exc)
            return ""

        try:
            text_after, _ = self.parse_html(
                self._extract_container_html(number), self.task_url
            )
        except ScraperError:
            return ""

        if text_after.startswith(text_before):
            return text_after[len(text_before) :].strip()
        return "" if text_after == text_before else text_after[len(text_before) :].strip()

    def scrape_browser(self, number: str) -> ScrapedTask:
        """Разобрать задачу через браузер (резервный путь).

        :raises ScraperError: задачу не удалось открыть или разобрать.
        """
        self.open_browser()
        self._throttle()
        self._submit_number(number)

        text, refs = self.parse_html(
            self._extract_container_html(number), self.task_url
        )
        if not text:
            raise ScraperError(f"Пустой текст условия у задачи {number}")

        answer = self._reveal_answer(number, text) if self.fetch_answer else ""

        return ScrapedTask(
            task_id=number,
            raw_text=text,
            url=self.task_url,
            site_answer=answer,
            attachments=self._collect_attachments(refs, number),
            source="browser",
        )

    # -- разбор HTML (общий для обоих источников) ----------------------------- #

    def parse_html(
        self, html: str, base_url: str
    ) -> tuple[str, list[tuple[str, str, str]]]:
        """Превратить HTML в текст и список ссылок на вложения.

        Порядок важен: сначала собираем изображения (пока разметка цела), затем
        восстанавливаем формулы, затем схлопываем таблицы в Markdown и только
        потом вытаскиваем текст.

        :returns: пара ``(текст, [(абсолютный_url, имя_файла, alt), ...])``.
        """
        soup = BeautifulSoup(html, "lxml")

        refs: list[tuple[str, str, str]] = []
        seen: set[str] = set()
        for tag in soup.select(self.selectors.images):
            raw_src = _pick_image_url(tag)
            if not raw_src or raw_src.startswith("data:"):
                continue
            absolute = urljoin(base_url, raw_src)
            if absolute in seen:
                continue
            seen.add(absolute)
            alt = tag.get("alt") or ""
            refs.append(
                (
                    absolute,
                    Path(urlparse(absolute).path).name,
                    alt.strip() if isinstance(alt, str) else "",
                )
            )

        _replace_math_nodes(soup)

        for selector in self.selectors.drop_css:
            for tag in soup.select(selector):
                tag.decompose()

        for tag in soup.find_all(True):
            if tag.get_text(strip=True) in self.selectors.drop_text:
                tag.decompose()

        for table in soup.find_all("table"):
            markdown = _table_to_markdown(table)
            table.replace_with(NavigableString(f"\n\n{markdown}\n\n"))

        return _normalize_whitespace(soup.get_text("\n")), refs

    # -- вложения -------------------------------------------------------------- #

    def download_file(self, url: str, destination: Path) -> Path | None:
        """Скачать один файл вложения.

        :returns: путь к файлу или ``None``, если скачать не удалось.
        """
        if destination.exists() and destination.stat().st_size > 0:
            logger.debug("Пропускаю уже скачанное %s", destination)
            return destination

        try:
            with self.session.get(url, timeout=self.timeout, stream=True) as response:
                response.raise_for_status()
                destination.parent.mkdir(parents=True, exist_ok=True)
                written = 0
                with destination.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=64 * 1024):
                        written += len(chunk)
                        if written > MAX_FILE_BYTES:
                            raise ScraperError(
                                f"Файл больше {MAX_FILE_BYTES} байт: {url}"
                            )
                        handle.write(chunk)
        except (requests.RequestException, ScraperError, OSError) as exc:
            logger.warning("Не удалось скачать %s: %s", url, exc)
            destination.unlink(missing_ok=True)
            return None
        return destination

    def _collect_attachments(
        self, refs: Iterable[tuple[str, str, str]], task_id: str
    ) -> list[Attachment]:
        """Скачать вложения задачи и извлечь содержимое таблиц и документов.

        Имя файла на диске включает номер задачи и исходное имя — иначе файлы
        разных задач сливаются в кучу, а по имени ``24-1.txt`` не понять, чьё оно.
        """
        if not self.download_attachments:
            return []

        results: list[Attachment] = []
        for index, (url, name, alt) in enumerate(refs, start=1):
            extension = _guess_extension(url, name)
            stem = Path(name).stem or str(index)
            filename = f"task_{task_id or 'unknown'}_{stem}{extension}"
            path = self.download_file(url, self.download_dir / filename)
            if path is None:
                continue

            attachment = Attachment(
                url=url,
                path=path,
                name=name or filename,
                kind=detect_kind(name or filename),
                alt=alt,
            )

            if self.extract_attachment_text and not attachment.is_image:
                try:
                    attachment.text, attachment.truncated = extract_text(
                        path, max_chars=self.max_attachment_chars
                    )
                except AttachmentError as exc:
                    # Нечитаемое вложение не повод терять задачу целиком:
                    # файл скачан, содержимое просто не попадёт в промпт.
                    logger.warning("Вложение %s не разобрано: %s", attachment.name, exc)

            results.append(attachment)
        return results

    # -- публичный API -------------------------------------------------------- #

    def scrape(self, number: str | int) -> ScrapedTask:
        """Получить задачу по номеру выбранным источником.

        :param number: номер задачи, как он показан на сайте (например, 21401).
        :raises ScraperError: задачу не удалось получить или разобрать.
        """
        number = str(number).strip()
        if not number:
            raise ScraperError("Пустой номер задачи")

        if self.source == "browser":
            task = self.scrape_browser(number)
        else:
            task = self.parse_payload(self.fetch_payload(number), number)

        logger.info(
            "Задача %s: %s символов, картинок %s, файлов данных %s, ответ сайта %s (%s)",
            task.task_id,
            len(task.raw_text),
            len(task.images),
            len(task.data_files),
            "есть" if task.site_answer else "нет",
            task.source,
        )
        return task

    def dump_page_html(self, number: str | int) -> str:
        """Вернуть HTML всей страницы с открытой задачей — для подбора селекторов."""
        self.open_browser()
        self._submit_number(str(number).strip())
        return self._page.content()


async def scrape_task_async(scraper: TaskScraper, number: str | int) -> ScrapedTask:
    """Асинхронная обёртка над :meth:`TaskScraper.scrape`.

    Скрапер синхронный, поэтому работа уходит в отдельный поток. В режиме
    ``browser`` экземпляр держит одну вкладку и не рассчитан на параллельные
    вызовы — заводите по экземпляру на поток либо сериализуйте обращения.
    """
    import asyncio

    return await asyncio.to_thread(scraper.scrape, number)


# --------------------------------------------------------------------------- #
# CLI — отладка
# --------------------------------------------------------------------------- #


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Разобрать аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description="Парсинг задачи kompege по номеру: условие, картинки, ответ."
    )
    parser.add_argument("number", help="Номер задачи, как на сайте (например, 21401)")
    parser.add_argument(
        "--source",
        choices=("api", "browser"),
        default="api",
        help="Откуда брать данные (по умолчанию api)",
    )
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help=f"База API (по умолчанию {DEFAULT_API_URL})",
    )
    parser.add_argument(
        "--task-url",
        default=DEFAULT_TASK_URL,
        help=f"Страница задачи для браузерного режима (по умолчанию {DEFAULT_TASK_URL})",
    )
    parser.add_argument(
        "--container",
        default=None,
        help="CSS-селектор блока задачи в браузерном режиме",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=DEFAULT_DOWNLOAD_DIR,
        help=f"Куда сохранять вложения (по умолчанию {DEFAULT_DOWNLOAD_DIR})",
    )
    parser.add_argument(
        "--no-attachments", action="store_true", help="Не скачивать вложения"
    )
    parser.add_argument(
        "--max-attachment-chars",
        type=int,
        default=DEFAULT_MAX_CHARS,
        help=f"Сколько символов содержимого файла оставлять (по умолчанию {DEFAULT_MAX_CHARS})",
    )
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="Показать текст, который уйдёт в LLM (условие + содержимое файлов)",
    )
    parser.add_argument(
        "--no-answer", action="store_true", help="Не забирать ответ с сайта"
    )
    parser.add_argument(
        "--headed", action="store_true", help="Показать окно браузера (отладка)"
    )
    parser.add_argument(
        "--dump-json",
        type=Path,
        default=None,
        help="Сохранить сырой ответ API и выйти (для уточнения ApiFieldConfig)",
    )
    parser.add_argument(
        "--dump-html",
        type=Path,
        default=None,
        help="Сохранить HTML страницы с открытой задачей и выйти (режим browser)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Точка входа CLI. Возвращает код возврата процесса."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = _parse_args(argv)

    scraper = TaskScraper(
        source=args.source,
        api_url=args.api_url,
        task_url=args.task_url,
        selectors=SelectorConfig(container_css=args.container),
        download_dir=args.download_dir,
        download_attachments=not args.no_attachments,
        max_attachment_chars=args.max_attachment_chars,
        fetch_answer=not args.no_answer,
        headless=not args.headed,
    )

    with scraper:
        try:
            if args.dump_json:
                payload = scraper.fetch_payload(args.number)
                args.dump_json.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                logger.info("JSON сохранён в %s", args.dump_json)
                return 0
            if args.dump_html:
                html = scraper.dump_page_html(args.number)
                args.dump_html.write_text(html, encoding="utf-8")
                logger.info("HTML сохранён в %s (%s байт)", args.dump_html, len(html))
                return 0
            task = scraper.scrape(args.number)
        except ScraperError as exc:
            logger.error("%s", exc)
            return 1

    print(task.build_prompt_text() if args.show_prompt else task.to_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
