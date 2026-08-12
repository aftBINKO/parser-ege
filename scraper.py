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
* изображения скачиваются локально;
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

load_dotenv()

logger = logging.getLogger(__name__)

#: База JSON API сайта.
DEFAULT_API_URL = os.getenv("KOMPEGE_API_URL", "https://kompege.ru/api/v1")

#: Страница поиска задачи — источник для резервного, браузерного режима.
DEFAULT_TASK_URL = os.getenv("KOMPEGE_TASK_URL", "https://kompege.ru/task")

#: Куда складывать скачанные изображения (папка в .gitignore).
DEFAULT_IMAGE_DIR = Path("downloads")

#: User-Agent обычного браузера: часть сайтов режет дефолтный UA requests.
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)

#: Скрипты матдвижков, блокируемые в браузерном режиме ради исходного LaTeX.
MATH_SCRIPT_PATTERNS = ("**/*mathjax*", "**/*MathJax*", "**/*katex*", "**/*KaTeX*")

#: Предохранитель от гигантских файлов при скачивании картинок, байты.
MAX_IMAGE_BYTES = 10 * 1024 * 1024


class ScraperError(Exception):
    """Не удалось получить или разобрать задачу."""


# --------------------------------------------------------------------------- #
# Конфигурация
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class ApiFieldConfig:
    """Имена полей в ответе API.

    Схема API заранее не известна, поэтому для каждого поля перечислены
    вероятные имена: берётся первое совпадение при обходе JSON в ширину (то
    есть самое «внешнее»). Когда посмотрите реальный ответ через ``--dump-json``,
    сведите каждый кортеж к единственному точному имени — так надёжнее.

    Все имена сравниваются в нижнем регистре.
    """

    condition: tuple[str, ...] = (
        "text", "condition", "statement", "task", "body", "question", "content", "html"
    )
    answer: tuple[str, ...] = ("answer", "correct_answer", "right_answer", "result")
    task_id: tuple[str, ...] = ("id", "number", "task_id", "num")
    images: tuple[str, ...] = ("images", "pictures", "files", "attachments", "media")
    solution: tuple[str, ...] = ("solution", "explanation", "analysis")


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
class DownloadedImage:
    """Скачанное изображение задачи."""

    url: str
    path: Path
    alt: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Словарь для сериализации (``Path`` приводится к строке)."""
        return {"url": self.url, "path": str(self.path), "alt": self.alt}


@dataclass(slots=True)
class ScrapedTask:
    """Результат парсинга одной задачи — вход для ``llm_processor.py``.

    :param site_answer: официальный ответ с сайта. Пустая строка, если сайт его
        не отдал. Не подставляется в ответ модели: это независимая величина для
        сверки.
    """

    task_id: str
    raw_text: str
    url: str = ""
    site_answer: str = ""
    images: list[DownloadedImage] = field(default_factory=list)
    source: str = "api"

    def to_dict(self) -> dict[str, Any]:
        """Словарь, пригодный для сериализации в JSON."""
        data = asdict(self)
        data["images"] = [image.to_dict() for image in self.images]
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

    Пустые ячейки сохраняются: в таблицах истинности пропуск — часть условия,
    и потерять его нельзя. Первая строка считается заголовком.
    """
    rows: list[list[str]] = []
    for row in table.find_all("tr"):
        cells = row.find_all(["th", "td"])
        if not cells:
            continue
        rows.append(
            [_normalize_whitespace(cell.get_text(" ")).replace("|", "\\|") for cell in cells]
        )

    if not rows:
        return ""

    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]

    lines = [
        "| " + " | ".join(rows[0]) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows[1:])
    return "\n".join(lines)


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

    Обход идёт в ширину, поэтому поле верхнего уровня выигрывает у одноимённого
    вложенного.

    :returns: значение строкой либо пустая строка, если ничего не нашлось.
    """
    wanted = {name.lower() for name in names}
    for node in _walk(payload):
        for key, value in node.items():
            if key.lower() in wanted and isinstance(value, (str, int, float)):
                text = str(value).strip()
                if text:
                    return text
    return ""


def find_image_urls(payload: Any, names: tuple[str, ...]) -> list[str]:
    """Собрать ссылки на изображения из списковых полей JSON.

    Понимает и список строк, и список объектов вида ``{"url": ...}``.
    """
    wanted = {name.lower() for name in names}
    urls: list[str] = []
    for node in _walk(payload):
        for key, value in node.items():
            if key.lower() not in wanted or not isinstance(value, list):
                continue
            for item in value:
                if isinstance(item, str) and item.strip():
                    urls.append(item.strip())
                elif isinstance(item, dict):
                    for candidate in ("url", "src", "path", "link", "file"):
                        found = item.get(candidate)
                        if isinstance(found, str) and found.strip():
                            urls.append(found.strip())
                            break
    return urls


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


def _guess_extension(url: str, content_type: str | None) -> str:
    """Подобрать расширение файла по URL, а при неудаче — по Content-Type."""
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".bmp"}:
        return suffix

    if content_type:
        guessed = mimetypes.guess_extension(content_type.split(";")[0].strip())
        if guessed:
            return ".jpg" if guessed == ".jpe" else guessed
    return ".png"


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
    :param image_dir: куда складывать картинки.
    :param download_images: скачивать ли изображения.
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
        image_dir: Path = DEFAULT_IMAGE_DIR,
        download_images: bool = True,
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
        self.image_dir = Path(image_dir)
        self.download_images = download_images
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
        режиме (таблицы, формулы, картинки).

        :raises ScraperError: в ответе не нашлось текста условия.
        """
        condition = find_field(payload, self.fields.condition)
        if not condition:
            raise ScraperError(
                f"В ответе API нет текста условия задачи {number}. "
                "Посмотрите JSON через --dump-json и уточните ApiFieldConfig."
            )

        images: list[tuple[str, str]] = []
        if _looks_like_html(condition):
            text, images = self.parse_html(condition, self.task_url)
        else:
            text = _normalize_whitespace(condition)

        for url in find_image_urls(payload, self.fields.images):
            absolute = urljoin(self.task_url, url)
            if absolute not in {existing for existing, _ in images}:
                images.append((absolute, ""))

        answer = find_field(payload, self.fields.answer) if self.fetch_answer else ""
        if self.fetch_answer and not answer:
            answer = self.fetch_answer_payload(number)
        if answer and _looks_like_html(answer):
            answer, _ = self.parse_html(answer, self.task_url)

        task_id = find_field(payload, self.fields.task_id) or number
        downloaded = self._download_all(images, number) if self.download_images else []

        return ScrapedTask(
            task_id=task_id,
            raw_text=text,
            url=f"{self.api_url}/task/{number}",
            site_answer=answer,
            images=downloaded,
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

        text, images = self.parse_html(
            self._extract_container_html(number), self.task_url
        )
        if not text:
            raise ScraperError(f"Пустой текст условия у задачи {number}")

        answer = self._reveal_answer(number, text) if self.fetch_answer else ""
        downloaded = self._download_all(images, number) if self.download_images else []

        return ScrapedTask(
            task_id=number,
            raw_text=text,
            url=self.task_url,
            site_answer=answer,
            images=downloaded,
            source="browser",
        )

    # -- разбор HTML (общий для обоих источников) ----------------------------- #

    def parse_html(self, html: str, base_url: str) -> tuple[str, list[tuple[str, str]]]:
        """Превратить HTML в текст и список ссылок на картинки.

        Порядок важен: сначала собираем изображения (пока разметка цела), затем
        восстанавливаем формулы, затем схлопываем таблицы в Markdown и только
        потом вытаскиваем текст.

        :returns: пара ``(текст, [(абсолютный_url_картинки, alt), ...])``.
        """
        soup = BeautifulSoup(html, "lxml")

        images: list[tuple[str, str]] = []
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
            images.append((absolute, alt.strip() if isinstance(alt, str) else ""))

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

        return _normalize_whitespace(soup.get_text("\n")), images

    # -- изображения ---------------------------------------------------------- #

    def download_image(self, image_url: str, destination: Path) -> Path | None:
        """Скачать одно изображение.

        :returns: путь к файлу или ``None``, если скачать не удалось.
        """
        if destination.exists() and destination.stat().st_size > 0:
            logger.debug("Пропускаю уже скачанное %s", destination)
            return destination

        try:
            with self.session.get(
                image_url, timeout=self.timeout, stream=True
            ) as response:
                response.raise_for_status()
                destination.parent.mkdir(parents=True, exist_ok=True)
                written = 0
                with destination.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=64 * 1024):
                        written += len(chunk)
                        if written > MAX_IMAGE_BYTES:
                            raise ScraperError(
                                f"Изображение больше {MAX_IMAGE_BYTES} байт: {image_url}"
                            )
                        handle.write(chunk)
        except (requests.RequestException, ScraperError, OSError) as exc:
            logger.warning("Не удалось скачать %s: %s", image_url, exc)
            destination.unlink(missing_ok=True)
            return None
        return destination

    def _download_all(
        self, images: Iterable[tuple[str, str]], task_id: str
    ) -> list[DownloadedImage]:
        """Скачать все картинки задачи, именуя их по номеру задачи и порядку."""
        results: list[DownloadedImage] = []
        for index, (image_url, alt) in enumerate(images, start=1):
            extension = _guess_extension(image_url, None)
            name = f"task_{task_id or 'unknown'}_{index}{extension}"
            path = self.download_image(image_url, self.image_dir / name)
            if path is not None:
                results.append(DownloadedImage(url=image_url, path=path, alt=alt))
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
            "Задача %s: %s символов, картинок %s, ответ сайта %s (%s)",
            task.task_id,
            len(task.raw_text),
            len(task.images),
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
        "--image-dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help=f"Куда сохранять картинки (по умолчанию {DEFAULT_IMAGE_DIR})",
    )
    parser.add_argument(
        "--no-images", action="store_true", help="Не скачивать изображения"
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
        image_dir=args.image_dir,
        download_images=not args.no_images,
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

    print(task.to_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
