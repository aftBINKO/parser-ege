"""Парсинг страницы задачи: текст условия + прикреплённые изображения.

Модуль — первый шаг пайплайна. На вход URL задачи, на выходе :class:`ScrapedTask`
с очищенным текстом и путями к скачанным картинкам; результат уходит дальше в
``llm_processor.py``.

Две стратегии получения HTML:

* ``static``  — ``requests``: быстро, дёшево, работает, если сервер отдаёт
  готовую разметку;
* ``dynamic`` — синхронный Playwright: нужен, если условие дорисовывается
  скриптами.

По умолчанию включён режим ``auto``: сначала пробуем ``requests``, и только если
контейнер задачи не нашёлся или текста подозрительно мало — поднимаем браузер.
Так в типичном случае не платим за запуск Chromium.

CSS-селекторы вынесены в :class:`SelectorConfig` — это плейсхолдеры, замените их
на реальные под разметку сайта. Чтобы посмотреть разметку, удобно сдампить
страницу::

    python scraper.py https://kompege.ru/task?id=123 --dump-html page.html

Модуль синхронный (``requests`` + sync Playwright — так стабильнее и проще
отлаживать), но для конкурентной обработки пачки задач есть асинхронная обёртка
:func:`scrape_task_async`, которая уводит работу в отдельный поток.
"""

from __future__ import annotations

import argparse
import json
import logging
import mimetypes
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qs, urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Tag
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

#: Куда складывать скачанные изображения (папка в .gitignore).
DEFAULT_IMAGE_DIR = Path("downloads")

#: User-Agent обычного браузера: часть сайтов режет дефолтный UA requests.
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)

#: Если статический HTML дал меньше символов — считаем, что контент рисует JS.
MIN_MEANINGFUL_TEXT_LEN = 40

#: Предохранитель от гигантских файлов при скачивании картинок, байты.
MAX_IMAGE_BYTES = 10 * 1024 * 1024


class ScraperError(Exception):
    """Не удалось получить или разобрать страницу задачи."""


# --------------------------------------------------------------------------- #
# Конфигурация селекторов
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class SelectorConfig:
    """CSS-селекторы элементов страницы задачи.

    ЗНАЧЕНИЯ НИЖЕ — ПЛЕЙСХОЛДЕРЫ. Подставьте реальные после осмотра разметки.

    :param container: корневой блок задачи; из него берётся текст и картинки.
    :param text: необязательный уточняющий селектор текста внутри контейнера.
        Если ``None`` — берётся весь текст контейнера.
    :param images: селектор изображений внутри контейнера.
    :param task_id: необязательный селектор элемента с ID задачи. Если ``None``
        или элемент не найден, ID достаётся из URL.
    :param drop: селекторы мусора, который нужно выкинуть из текста
        (кнопки, счётчики, блок «ответ» и т.п.).
    """

    container: str = ".task-content"
    text: str | None = None
    images: str = "img"
    task_id: str | None = None
    drop: tuple[str, ...] = ("script", "style", "noscript")


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
    """Результат парсинга одной страницы — вход для ``llm_processor.py``."""

    url: str
    task_id: str
    raw_text: str
    images: list[DownloadedImage] = field(default_factory=list)
    strategy: str = "static"

    def to_dict(self) -> dict[str, Any]:
        """Словарь, пригодный для сериализации в JSON."""
        data = asdict(self)
        data["images"] = [image.to_dict() for image in self.images]
        return data

    def to_json(self, *, indent: int = 2) -> str:
        """JSON-строка (UTF-8, без экранирования кириллицы)."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


# --------------------------------------------------------------------------- #
# Вспомогательные функции
# --------------------------------------------------------------------------- #


def _normalize_whitespace(text: str) -> str:
    """Схлопнуть пробелы и лишние переносы, убрать неразрывные пробелы."""
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_task_id_from_url(url: str) -> str:
    """Достать ID задачи из URL.

    Понимает и query-параметры (``?id=123``, ``?task_id=123``), и «красивые»
    пути (``/task/123``). Если ничего не нашлось — возвращает пустую строку.
    """
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    for key in ("id", "task_id", "taskId", "number"):
        values = query.get(key)
        if values and values[0].strip():
            return values[0].strip()

    for segment in reversed([part for part in parsed.path.split("/") if part]):
        if segment.isdigit():
            return segment
    return ""


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


# --------------------------------------------------------------------------- #
# Скрапер
# --------------------------------------------------------------------------- #


class TaskScraper:
    """Загружает и разбирает страницы задач.

    :param selectors: конфигурация CSS-селекторов.
    :param strategy: ``auto`` | ``static`` | ``dynamic``.
    :param image_dir: куда складывать картинки.
    :param download_images: скачивать ли изображения (``False`` — только ссылки).
    :param timeout: таймаут HTTP-запроса, сек.
    :param request_delay: пауза между запросами, сек — вежливость к сайту.
    :param wait_selector: что ждать в динамическом режиме; по умолчанию —
        ``selectors.container``.
    """

    def __init__(
        self,
        selectors: SelectorConfig | None = None,
        *,
        strategy: str = "auto",
        image_dir: Path = DEFAULT_IMAGE_DIR,
        download_images: bool = True,
        timeout: float = 20.0,
        request_delay: float = 1.0,
        wait_selector: str | None = None,
    ) -> None:
        if strategy not in {"auto", "static", "dynamic"}:
            raise ValueError(f"Неизвестная стратегия: {strategy}")

        self.selectors = selectors or SelectorConfig()
        self.strategy = strategy
        self.image_dir = Path(image_dir)
        self.download_images = download_images
        self.timeout = timeout
        self.request_delay = request_delay
        self.wait_selector = wait_selector or self.selectors.container

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": DEFAULT_USER_AGENT})
        self._last_request_at = 0.0

    # -- получение HTML ------------------------------------------------------ #

    def _throttle(self) -> None:
        """Выдержать паузу между запросами к сайту."""
        elapsed = time.monotonic() - self._last_request_at
        if self._last_request_at and elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self._last_request_at = time.monotonic()

    def fetch_static(self, url: str) -> str:
        """Забрать HTML через ``requests``.

        :raises ScraperError: сетевая ошибка или не-2xx ответ.
        """
        self._throttle()
        logger.debug("GET %s", url)
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise ScraperError(f"Не удалось загрузить {url}: {exc}") from exc

        # requests иногда ошибается с кодировкой кириллицы, если её нет в заголовках
        if response.encoding and response.encoding.lower() == "iso-8859-1":
            response.encoding = response.apparent_encoding
        return response.text

    def fetch_dynamic(self, url: str) -> str:
        """Забрать HTML через headless-браузер (Playwright).

        Импорт локальный: если сайт статический, playwright можно не ставить.

        :raises ScraperError: браузер не поднялся или контент не дождались.
        """
        try:
            from playwright.sync_api import Error as PlaywrightError
            from playwright.sync_api import sync_playwright
        except ImportError as exc:  # pragma: no cover - зависит от окружения
            raise ScraperError(
                "Для динамического режима нужен playwright: pip install playwright"
            ) from exc

        self._throttle()
        logger.debug("Открываю %s в headless-браузере", url)
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(user_agent=DEFAULT_USER_AGENT)
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=self.timeout * 1000)
                try:
                    page.wait_for_selector(
                        self.wait_selector, timeout=self.timeout * 1000
                    )
                except PlaywrightError:
                    # Не фатально: возможно, селектор-плейсхолдер ещё не заменён.
                    logger.warning(
                        "Селектор '%s' не дождались на %s — разбираю что есть",
                        self.wait_selector,
                        url,
                    )
                return page.content()
            except PlaywrightError as exc:
                raise ScraperError(f"Playwright не смог открыть {url}: {exc}") from exc
            finally:
                browser.close()

    # -- разбор -------------------------------------------------------------- #

    def _find_container(self, soup: BeautifulSoup) -> Tag | None:
        """Найти корневой блок задачи."""
        return soup.select_one(self.selectors.container)

    def parse_html(self, html: str, url: str) -> tuple[str, list[tuple[str, str]]]:
        """Вытащить из HTML текст условия и ссылки на изображения.

        :returns: пара ``(текст, [(абсолютный_url_картинки, alt), ...])``.
        :raises ScraperError: контейнер задачи не найден.
        """
        soup = BeautifulSoup(html, "lxml")
        container = self._find_container(soup)
        if container is None:
            raise ScraperError(
                f"Контейнер '{self.selectors.container}' не найден на {url}. "
                "Проверьте селекторы в SelectorConfig."
            )

        # Картинки собираем до чистки текста, чтобы ничего не потерять.
        images: list[tuple[str, str]] = []
        seen: set[str] = set()
        for tag in container.select(self.selectors.images):
            raw_src = _pick_image_url(tag)
            if not raw_src or raw_src.startswith("data:"):
                continue
            absolute = urljoin(url, raw_src)
            if absolute in seen:
                continue
            seen.add(absolute)
            alt = tag.get("alt") or ""
            images.append((absolute, alt.strip() if isinstance(alt, str) else ""))

        for selector in self.selectors.drop:
            for tag in container.select(selector):
                tag.decompose()

        text_node = container
        if self.selectors.text:
            found = container.select_one(self.selectors.text)
            if found is not None:
                text_node = found

        text = _normalize_whitespace(text_node.get_text("\n"))
        return text, images

    def _resolve_task_id(self, html: str, url: str) -> str:
        """Определить ID задачи: сначала по селектору, потом по URL."""
        if self.selectors.task_id:
            node = BeautifulSoup(html, "lxml").select_one(self.selectors.task_id)
            if node is not None:
                found = _normalize_whitespace(node.get_text(" "))
                digits = re.search(r"\d+", found)
                if digits:
                    return digits.group(0)
        return extract_task_id_from_url(url)

    # -- изображения --------------------------------------------------------- #

    def download_image(self, image_url: str, destination: Path) -> Path | None:
        """Скачать одно изображение.

        :returns: путь к файлу или ``None``, если скачать не удалось.
        """
        if destination.exists() and destination.stat().st_size > 0:
            logger.debug("Пропускаю уже скачанное %s", destination)
            return destination

        self._throttle()
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
        """Скачать все картинки задачи, именуя их по ID задачи и порядку."""
        results: list[DownloadedImage] = []
        for index, (image_url, alt) in enumerate(images, start=1):
            extension = _guess_extension(image_url, None)
            name = f"task_{task_id or 'unknown'}_{index}{extension}"
            path = self.download_image(image_url, self.image_dir / name)
            if path is not None:
                results.append(DownloadedImage(url=image_url, path=path, alt=alt))
        return results

    # -- публичный API ------------------------------------------------------- #

    def scrape(self, url: str) -> ScrapedTask:
        """Разобрать страницу задачи целиком.

        В режиме ``auto`` при неудаче статического разбора автоматически
        переключается на браузер.

        :raises ScraperError: страницу не удалось загрузить или разобрать.
        """
        used_strategy = self.strategy
        html = ""
        text = ""
        images: list[tuple[str, str]] = []

        if self.strategy in {"auto", "static"}:
            html = self.fetch_static(url)
            try:
                text, images = self.parse_html(html, url)
            except ScraperError:
                if self.strategy == "static":
                    raise
                text = ""

            if self.strategy == "auto" and len(text) < MIN_MEANINGFUL_TEXT_LEN:
                logger.info(
                    "Статический HTML %s дал %s символов — пробую браузер",
                    url,
                    len(text),
                )
                html = self.fetch_dynamic(url)
                text, images = self.parse_html(html, url)
                used_strategy = "dynamic"
            else:
                used_strategy = "static"
        else:
            html = self.fetch_dynamic(url)
            text, images = self.parse_html(html, url)
            used_strategy = "dynamic"

        if not text:
            raise ScraperError(f"Пустой текст условия на {url}")

        task_id = self._resolve_task_id(html, url)
        downloaded = self._download_all(images, task_id) if self.download_images else []

        logger.info(
            "Задача %s: %s символов, картинок %s (%s)",
            task_id or "<без id>",
            len(text),
            len(downloaded),
            used_strategy,
        )
        return ScrapedTask(
            url=url,
            task_id=task_id,
            raw_text=text,
            images=downloaded,
            strategy=used_strategy,
        )

    def close(self) -> None:
        """Закрыть HTTP-сессию."""
        self.session.close()

    def __enter__(self) -> "TaskScraper":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


async def scrape_task_async(scraper: TaskScraper, url: str) -> ScrapedTask:
    """Асинхронная обёртка над :meth:`TaskScraper.scrape`.

    Сам скрапер синхронный, поэтому работа уходит в отдельный поток — это даёт
    ``main.py`` возможность собирать задачи конкурентно через ``asyncio.gather``,
    не переписывая модуль на aiohttp.
    """
    import asyncio

    return await asyncio.to_thread(scraper.scrape, url)


# --------------------------------------------------------------------------- #
# CLI — отладка селекторов
# --------------------------------------------------------------------------- #


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Разобрать аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description="Парсинг страницы задачи: текст условия и изображения."
    )
    parser.add_argument("url", help="URL страницы задачи")
    parser.add_argument(
        "--strategy",
        choices=("auto", "static", "dynamic"),
        default="auto",
        help="Способ получения HTML (по умолчанию auto)",
    )
    parser.add_argument(
        "--container",
        default=SelectorConfig.container,
        help="CSS-селектор блока с задачей",
    )
    parser.add_argument(
        "--images-selector",
        default=SelectorConfig.images,
        help="CSS-селектор изображений внутри блока",
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
        "--dump-html",
        type=Path,
        default=None,
        help="Сохранить сырой HTML страницы в файл (для подбора селекторов)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Точка входа CLI. Возвращает код возврата процесса."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = _parse_args(argv)

    selectors = SelectorConfig(
        container=args.container, images=args.images_selector
    )
    scraper = TaskScraper(
        selectors,
        strategy=args.strategy,
        image_dir=args.image_dir,
        download_images=not args.no_images,
    )

    with scraper:
        if args.dump_html:
            html = (
                scraper.fetch_dynamic(args.url)
                if args.strategy == "dynamic"
                else scraper.fetch_static(args.url)
            )
            args.dump_html.write_text(html, encoding="utf-8")
            logger.info("HTML сохранён в %s (%s байт)", args.dump_html, len(html))
            return 0

        try:
            task = scraper.scrape(args.url)
        except ScraperError as exc:
            logger.error("%s", exc)
            return 1

    print(task.to_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
