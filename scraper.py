"""Парсинг задачи с kompege.ru: текст условия, формулы, таблицы, изображения.

Особенности сайта, определившие устройство модуля:

* URL один на все задачи — ``https://kompege.ru/task``. Конкретная задача
  открывается через форму «Поиск по номеру», поэтому входом пайплайна служит
  **номер задачи**, а не ссылка. Отсюда же необходимость браузера: получить
  условие простым GET нельзя.
* Формулы отрисовывает математический движок (MathJax/KaTeX). Его загрузка по
  умолчанию блокируется — тогда в DOM остаётся исходный LaTeX, который куда
  полезнее для LLM, чем текст отрендеренных глифов (``∧``, ``¬``). См.
  ``block_math_js``.
* Условие часто содержит таблицы (например, таблицы истинности). Они
  конвертируются в Markdown: плоская строка цифр для модели бесполезна, а
  пустые ячейки в таких задачах значимы.

Браузер поднимается один раз на весь прогон и переиспользуется между задачами —
это на порядок быстрее, чем запускать Chromium на каждый номер::

    with TaskScraper() as scraper:
        for number in ("21401", "21402"):
            task = scraper.scrape(number)

Playwright используется синхронный: так стабильнее и проще отлаживать. Для
конкурентной работы из ``main.py`` есть обёртка :func:`scrape_task_async`.

Отладка селекторов на реальной разметке::

    python scraper.py 21401 --dump-html page.html
"""

from __future__ import annotations

import argparse
import json
import logging
import mimetypes
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, NavigableString, Tag
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

#: Страница поиска задачи (одна на все задачи).
DEFAULT_TASK_URL = os.getenv("KOMPEGE_TASK_URL", "https://kompege.ru/task")

#: Куда складывать скачанные изображения (папка в .gitignore).
DEFAULT_IMAGE_DIR = Path("downloads")

#: User-Agent обычного браузера: часть сайтов режет дефолтный UA requests.
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)

#: Скрипты матдвижков, которые блокируются ради исходного LaTeX в DOM.
MATH_SCRIPT_PATTERNS = ("**/*mathjax*", "**/*MathJax*", "**/*katex*", "**/*KaTeX*")

#: Предохранитель от гигантских файлов при скачивании картинок, байты.
MAX_IMAGE_BYTES = 10 * 1024 * 1024


class ScraperError(Exception):
    """Не удалось получить или разобрать задачу."""


# --------------------------------------------------------------------------- #
# Конфигурация селекторов
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class SelectorConfig:
    """Как найти элементы на странице задачи.

    Поиск формы намеренно идёт по видимому тексту, а не по CSS-классам: подпись
    «Номер задачи» стабильнее сгенерированных классов. При необходимости всё
    переопределяется.

    :param number_input_placeholder: placeholder поля ввода номера.
    :param submit_button_text: текст кнопки отправки формы.
    :param container_css: CSS-селектор блока с задачей. Если ``None`` — блок
        ищется эвристикой по заголовку вида «№ 21401» (см.
        :meth:`TaskScraper._extract_container_html`). Задайте реальный селектор,
        когда посмотрите разметку — это надёжнее эвристики.
    :param images: селектор изображений внутри блока задачи.
    :param drop_css: селекторы мусора, удаляемого из текста.
    :param drop_text: элементы, у которых текст точно совпадает с одной из этих
        строк, удаляются целиком (кнопки-ссылки интерфейса).
    """

    number_input_placeholder: str = "Номер задачи"
    submit_button_text: str = "Показать задачу"
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
    """Результат парсинга одной задачи — вход для ``llm_processor.py``."""

    task_id: str
    raw_text: str
    url: str = DEFAULT_TASK_URL
    images: list[DownloadedImage] = field(default_factory=list)

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
    """Вернуть формулам исходный LaTeX там, где движок уже успел отработать.

    Основной путь — блокировка матскрипта (тогда LaTeX и так остаётся в DOM),
    это подстраховка: MathJax v2 сохраняет исходник в ``<script type="math/tex">``,
    а v3 и KaTeX прячут его в ``annotation[encoding="application/x-tex"]``.
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
            if parent.name in {"mjx-container", "math", "span"} and "katex" in " ".join(
                parent.get("class", [])
            ):
                target = parent
            elif parent.name == "mjx-container":
                target = parent
                break
        target.replace_with(NavigableString(f" ${latex}$ " if latex else " "))


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


#: JS-эвристика: найти блок задачи по заголовку «№ <номер>».
#: Берётся самый глубокий элемент с этим текстом, затем подъём вверх, пока блок
#: не наберёт осмысленный объём — так в выборку попадает всё условие целиком.
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
    """Открывает задачи по номеру через форму сайта и разбирает их разметку.

    :param selectors: конфигурация поиска элементов.
    :param task_url: страница с формой поиска задачи.
    :param image_dir: куда складывать картинки.
    :param download_images: скачивать ли изображения (``False`` — только ссылки).
    :param headless: запускать браузер без окна.
    :param block_math_js: блокировать матдвижок, чтобы получить исходный LaTeX.
    :param timeout: таймаут ожиданий Playwright, сек.
    :param request_delay: пауза между задачами, сек — вежливость к сайту.
    :param min_container_length: порог для эвристики поиска блока задачи.
    """

    def __init__(
        self,
        selectors: SelectorConfig | None = None,
        *,
        task_url: str = DEFAULT_TASK_URL,
        image_dir: Path = DEFAULT_IMAGE_DIR,
        download_images: bool = True,
        headless: bool = True,
        block_math_js: bool = True,
        timeout: float = 30.0,
        request_delay: float = 1.0,
        min_container_length: int = 200,
    ) -> None:
        self.selectors = selectors or SelectorConfig()
        self.task_url = task_url
        self.image_dir = Path(image_dir)
        self.download_images = download_images
        self.headless = headless
        self.block_math_js = block_math_js
        self.timeout_ms = int(timeout * 1000)
        self.request_delay = request_delay
        self.min_container_length = min_container_length

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": DEFAULT_USER_AGENT})
        self._last_request_at = 0.0

        self._playwright: Any = None
        self._browser: Any = None
        self._page: Any = None

    # -- жизненный цикл браузера --------------------------------------------- #

    def open(self) -> None:
        """Поднять браузер и открыть страницу поиска задач.

        Вызывается лениво из :meth:`scrape`, но можно и явно.

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
            self._page.set_default_timeout(self.timeout_ms)

            if self.block_math_js:
                # Без матдвижка формулы остаются в DOM исходным LaTeX.
                for pattern in MATH_SCRIPT_PATTERNS:
                    self._page.route(pattern, lambda route: route.abort())

            logger.debug("Открываю %s", self.task_url)
            self._page.goto(self.task_url, wait_until="domcontentloaded")
        except PlaywrightError as exc:
            self.close()
            raise ScraperError(f"Не удалось открыть {self.task_url}: {exc}") from exc

    def close(self) -> None:
        """Закрыть браузер и HTTP-сессию."""
        for resource, name in ((self._browser, "browser"), (self._playwright, "playwright")):
            if resource is None:
                continue
            try:
                resource.stop() if name == "playwright" else resource.close()
            except Exception as exc:  # pragma: no cover - гасим шум при выходе
                logger.debug("Ошибка при закрытии %s: %s", name, exc)
        self._playwright = self._browser = self._page = None
        self.session.close()

    def __enter__(self) -> "TaskScraper":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- получение разметки задачи ------------------------------------------- #

    def _throttle(self) -> None:
        """Выдержать паузу между обращениями к сайту."""
        elapsed = time.monotonic() - self._last_request_at
        if self._last_request_at and elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self._last_request_at = time.monotonic()

    def _submit_number(self, number: str) -> None:
        """Ввести номер задачи в форму и отправить её.

        :raises ScraperError: форма не найдена или задача не появилась.
        """
        from playwright.sync_api import Error as PlaywrightError

        page = self._page
        try:
            field = page.get_by_placeholder(self.selectors.number_input_placeholder)
            field.wait_for(state="visible")
            field.fill(number)
            page.get_by_role(
                "button", name=self.selectors.submit_button_text
            ).first.click()

            # Задача считается загруженной, когда на странице появился её номер.
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

    # -- разбор -------------------------------------------------------------- #

    def parse_html(self, html: str, base_url: str) -> tuple[str, list[tuple[str, str]]]:
        """Превратить HTML блока задачи в текст и список ссылок на картинки.

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

    # -- изображения --------------------------------------------------------- #

    def download_image(self, image_url: str, destination: Path) -> Path | None:
        """Скачать одно изображение.

        :returns: путь к файлу или ``None``, если скачать не удалось.
        """
        if destination.exists() and destination.stat().st_size > 0:
            logger.debug("Пропускаю уже скачанное %s", destination)
            return destination

        try:
            with self.session.get(image_url, timeout=30, stream=True) as response:
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

    # -- публичный API ------------------------------------------------------- #

    def scrape(self, number: str | int) -> ScrapedTask:
        """Открыть задачу по номеру и разобрать её.

        :param number: номер задачи, как он показан на сайте (например, 21401).
        :raises ScraperError: задачу не удалось открыть или разобрать.
        """
        number = str(number).strip()
        if not number:
            raise ScraperError("Пустой номер задачи")

        self.open()
        self._throttle()
        self._submit_number(number)

        html = self._extract_container_html(number)
        text, images = self.parse_html(html, self.task_url)
        if not text:
            raise ScraperError(f"Пустой текст условия у задачи {number}")

        downloaded = self._download_all(images, number) if self.download_images else []
        logger.info(
            "Задача %s: %s символов, картинок %s", number, len(text), len(downloaded)
        )
        return ScrapedTask(
            task_id=number, raw_text=text, url=self.task_url, images=downloaded
        )

    def dump_page_html(self, number: str | int) -> str:
        """Вернуть HTML всей страницы с открытой задачей — для подбора селекторов."""
        self.open()
        self._submit_number(str(number).strip())
        return self._page.content()


async def scrape_task_async(scraper: TaskScraper, number: str | int) -> ScrapedTask:
    """Асинхронная обёртка над :meth:`TaskScraper.scrape`.

    Скрапер синхронный, поэтому работа уходит в отдельный поток. Внимание: один
    экземпляр :class:`TaskScraper` держит одну вкладку и не рассчитан на
    параллельные вызовы — на каждый поток заводите свой экземпляр либо
    сериализуйте обращения семафором на единицу.
    """
    import asyncio

    return await asyncio.to_thread(scraper.scrape, number)


# --------------------------------------------------------------------------- #
# CLI — отладка селекторов
# --------------------------------------------------------------------------- #


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Разобрать аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description="Парсинг задачи kompege по номеру: условие и изображения."
    )
    parser.add_argument("number", help="Номер задачи, как на сайте (например, 21401)")
    parser.add_argument(
        "--task-url",
        default=DEFAULT_TASK_URL,
        help=f"Страница поиска задачи (по умолчанию {DEFAULT_TASK_URL})",
    )
    parser.add_argument(
        "--container",
        default=None,
        help="CSS-селектор блока задачи (по умолчанию — эвристика по номеру)",
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
        "--headed", action="store_true", help="Показать окно браузера (отладка)"
    )
    parser.add_argument(
        "--render-math",
        action="store_true",
        help="Не блокировать матдвижок (формулы придут отрендеренными, не LaTeX)",
    )
    parser.add_argument(
        "--dump-html",
        type=Path,
        default=None,
        help="Сохранить HTML всей страницы с открытой задачей и выйти",
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
        SelectorConfig(container_css=args.container),
        task_url=args.task_url,
        image_dir=args.image_dir,
        download_images=not args.no_images,
        headless=not args.headed,
        block_math_js=not args.render_math,
    )

    with scraper:
        try:
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
