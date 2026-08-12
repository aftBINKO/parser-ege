"""Публикация готовой задачи в админ-панель сайта.

Модуль — последний шаг пайплайна: берёт :class:`~llm_processor.TaskSolution`,
открывает страницу создания задачи в админке под сохранённой сессией и
раскладывает данные по полям формы.

Сессия берётся из ``auth_state.json`` (см. ``auth_state.py``) — логиниться при
каждом запуске не нужно. Если сессия протухла, модуль это замечает и говорит
прямо, вместо того чтобы молча заполнять форму на странице логина.

Playwright используется синхронный: сценарий линейный, отлаживать его по шагам
проще, а выигрыш от асинхронности здесь нулевой — узкое место не в CPU.

**Селекторы в :class:`FieldSelectors` — плейсхолдеры.** Замените их на реальные,
подсмотрев разметку своей админки. Модуль умеет работать и с обычными
``input``/``textarea``, и с визуальными редакторами (contenteditable), и с
редакторами внутри iframe (TinyMCE и подобные) — см. :class:`FieldSelectors`.

Перед первой боевой публикацией прогоните вхолостую: форма заполнится, но
кнопка сохранения нажата не будет::

    python uploader.py solution.json --dry-run --headed
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from dotenv import load_dotenv

from formatting import render_hints, render_plain
from llm_processor import TaskSolution

load_dotenv()

logger = logging.getLogger(__name__)

#: Страница создания задачи в админке.
DEFAULT_CREATE_URL = os.getenv("ADMIN_CREATE_TASK_URL", "")

#: Файл с сохранённой сессией (создаётся auth_state.py).
DEFAULT_STATE_PATH = Path(os.getenv("AUTH_STATE_PATH", "auth_state.json"))

#: Признаки того, что нас выкинуло на страницу логина.
LOGIN_URL_MARKERS = ("login", "signin", "sign-in", "auth")

#: JS для ``--inspect``: собирает поля ввода вместе с готовыми селекторами.
#: Подпись ищется по ``<label for>``, затем по родительскому ``<label>``, затем
#: по ближайшему тексту выше — в админках встречаются все три варианта.
_INSPECT_JS = """
() => {
  const cssEscape = (value) =>
    window.CSS && CSS.escape ? CSS.escape(value) : value.replace(/([^\\w-])/g, '\\\\$1');

  const labelFor = (el) => {
    if (el.id) {
      const byFor = document.querySelector(`label[for="${el.id}"]`);
      if (byFor) return byFor.innerText.trim();
    }
    const parent = el.closest('label');
    if (parent) return parent.innerText.trim();
    // Внутри обёртки визуального редактора «ближайшим текстом» окажутся
    // подписи кнопок панели — подписью поля они не являются.
    if (el.closest('.fr-box, .tox-tinymce, .cke, .ql-container')) return '';
    const wrapper = el.closest('div, td, li, fieldset');
    if (wrapper) {
      const text = (wrapper.innerText || '').trim().split('\\n')[0];
      if (text && text.length < 80) return text;
    }
    return '';
  };

  const selectorFor = (el, index) => {
    if (el.id) return '#' + cssEscape(el.id);
    if (el.name) return `${el.tagName.toLowerCase()}[name="${el.name}"]`;
    const cls = (el.className || '').toString().trim().split(/\\s+/).filter(Boolean)[0];
    if (cls) return `${el.tagName.toLowerCase()}.${cssEscape(cls)}`;
    return `${el.tagName.toLowerCase()}:nth-of-type(${index + 1})`;
  };

  const nodes = document.querySelectorAll(
    'input, textarea, select, [contenteditable="true"], button, [type="submit"]'
  );

  // Панели визуальных редакторов (Froala, TinyMCE, CKEditor) дают сотни кнопок
  // «Bold», «Italic» и подобных. К форме они отношения не имеют и только
  // топят полезные поля — помечаем их, чтобы отфильтровать при выводе.
  const isEditorChrome = (el) =>
    Boolean(el.closest('.fr-toolbar, .fr-popup, .tox-toolbar, .tox-tbtn, .cke_toolbox, .ql-toolbar'));

  return Array.from(nodes).map((el, index) => {
    const tag = el.tagName.toLowerCase();
    const type = (el.getAttribute('type') || '').toLowerCase();
    const editable = el.getAttribute('contenteditable') === 'true';
    const box = el.getBoundingClientRect();
    return {
      selector: selectorFor(el, index),
      tag: tag,
      type: type,
      id: el.id || '',
      name: el.getAttribute('name') || '',
      placeholder: el.getAttribute('placeholder') || '',
      label: labelFor(el),
      text: tag === 'button' || type === 'submit' ? (el.innerText || el.value || '').trim() : '',
      rich: editable,
      visible: box.width > 0 && box.height > 0,
      chrome: isEditorChrome(el),
    };
  }).filter((item) => item.tag !== 'input' || !['hidden'].includes(item.type));
}
"""


class UploaderError(Exception):
    """Базовая ошибка публикации."""


class AuthStateError(UploaderError):
    """Сессия отсутствует или больше не действует — нужен ``auth_state.py``."""


# --------------------------------------------------------------------------- #
# Конфигурация
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class FieldSelectors:
    """CSS-селекторы полей формы создания задачи.

    Значения подставлены по выводу ``--inspect`` на реальной админке и требуют
    проверки холостым прогоном. Пустая строка означает «этого поля в форме нет»
    — шаг просто пропускается.

    Три замечания по этой конкретной админке:

    * условие, решение и доп. текст — редакторы Froala. Они не ``textarea``, а
      ``div.fr-element``, и на странице их три подряд, поэтому различаются по
      порядковому номеру (``>> nth=``). Порядок соответствует подписям
      «Вопрос», «Решение», «Доп. текст»;
    * ``#title`` — это «Максимальный балл», а условие лежит в
      ``textarea[name="title"]``. Ловушка: по имени легко перепутать;
    * кнопок ``button.button`` две («Сохранить» и «Пересчитать баллы»), поэтому
      сохранение ищется по тексту.

    :param task_id: поле с номером задачи. В этой админке отдельного поля под
        номер задачи с kompege нет — оставлено пустым.
    :param condition: поле условия («Вопрос»).
    :param solution: поле с текстом решения.
    :param answer: поле ответа.
    :param hints: поле подсказок (все подсказки склеиваются в одну строку).
    :param save_button: кнопка сохранения.
    :param success_indicator: элемент, появляющийся после успешного сохранения.
        Если пусто — успех определяется по отсутствию ошибок и тому, что
        страница устоялась.
    :param error_indicator: элемент с сообщением об ошибке формы.
    :param rich_text_fields: поля-редакторы: вместо ``fill()`` в них печатают
        как в ``contenteditable``. Для Froala это обязательно — иначе редактор
        не заметит вставку и не перенесёт её в скрытую ``textarea`` при
        сохранении.
    :param editor_frames: поля, чей редактор живёт внутри iframe (TinyMCE и
        подобные): ``{"condition": "iframe#condition_ifr"}``.
    :param selects: выпадающие списки, которые нужно выставить перед
        сохранением: ``{селектор: значение}``. Значение ищется по видимому
        тексту пункта, а если такого нет — по его ``value``.
    """

    task_id: str = ""
    condition: str = "div.fr-element >> nth=0"
    solution: str = "div.fr-element >> nth=1"
    answer: str = ""
    hints: str = ""
    save_button: str = 'button:has-text("Сохранить")'
    success_indicator: str = ""
    error_indicator: str = ".errorlist, .alert-danger, .invalid-feedback"
    rich_text_fields: tuple[str, ...] = ("condition", "solution")
    editor_frames: dict[str, str] = field(default_factory=dict)
    selects: dict[str, str] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Результат
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class UploadResult:
    """Итог публикации одной задачи."""

    task_id: str
    ok: bool
    message: str = ""
    screenshot: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        """Словарь для сериализации/лога."""
        return {
            "task_id": self.task_id,
            "ok": self.ok,
            "message": self.message,
            "screenshot": str(self.screenshot) if self.screenshot else None,
        }


# --------------------------------------------------------------------------- #
# Загрузчик
# --------------------------------------------------------------------------- #


class AdminUploader:
    """Публикует задачи в админке под сохранённой сессией.

    :param create_url: страница создания задачи.
    :param state_path: файл сессии, созданный ``auth_state.py``.
    :param selectors: селекторы полей формы.
    :param headless: запускать браузер без окна. При отладке селекторов удобнее
        ``False`` — видно, куда именно попадают данные.
    :param dry_run: заполнить форму, но не нажимать сохранение.
    :param timeout: таймаут ожидания элементов, сек.
    :param hints_separator: чем склеивать подсказки в одно поле.
    :param render: очищать текст от Markdown и LaTeX перед вставкой. Выключайте,
        только если поле админки само разбирает эту разметку.
    :param screenshot_dir: куда класть скриншоты неудачных публикаций. ``None`` —
        не делать скриншоты.
    """

    def __init__(
        self,
        *,
        create_url: str = DEFAULT_CREATE_URL,
        state_path: Path = DEFAULT_STATE_PATH,
        selectors: FieldSelectors | None = None,
        headless: bool = True,
        dry_run: bool = False,
        timeout: float = 30.0,
        hints_separator: str = "\n",
        render: bool = True,
        screenshot_dir: Path | None = None,
    ) -> None:
        if not create_url:
            raise UploaderError(
                "Не задан URL создания задачи: укажите ADMIN_CREATE_TASK_URL в .env "
                "или передайте create_url"
            )

        self.create_url = create_url
        self.state_path = Path(state_path)
        self.selectors = selectors or FieldSelectors()
        self.headless = headless
        self.dry_run = dry_run
        self.timeout_ms = int(timeout * 1000)
        self.hints_separator = hints_separator
        self.render = render
        self.screenshot_dir = Path(screenshot_dir) if screenshot_dir else None

        self._playwright: Any = None
        self._browser: Any = None
        self._context: Any = None
        self._page: Any = None

    # -- жизненный цикл ------------------------------------------------------ #

    def open(self) -> None:
        """Поднять браузер с сохранённой сессией.

        :raises AuthStateError: файла сессии нет — сначала ``auth_state.py``.
        :raises UploaderError: playwright не установлен или браузер не поднялся.
        """
        if self._page is not None:
            return

        if not self.state_path.exists():
            raise AuthStateError(
                f"Файл сессии {self.state_path} не найден. "
                "Выполните: python auth_state.py"
            )

        try:
            from playwright.sync_api import Error as PlaywrightError
            from playwright.sync_api import sync_playwright
        except ImportError as exc:  # pragma: no cover - зависит от окружения
            raise UploaderError(
                "Нужен playwright: pip install playwright && playwright install chromium"
            ) from exc

        self._playwright = sync_playwright().start()
        try:
            self._browser = self._playwright.chromium.launch(headless=self.headless)
            self._context = self._browser.new_context(
                storage_state=str(self.state_path)
            )
            self._context.set_default_timeout(self.timeout_ms)
            self._page = self._context.new_page()
        except PlaywrightError as exc:
            self.close()
            raise UploaderError(f"Не удалось запустить браузер: {exc}") from exc

    def close(self) -> None:
        """Закрыть браузер."""
        for resource, name in (
            (self._context, "context"),
            (self._browser, "browser"),
            (self._playwright, "playwright"),
        ):
            if resource is None:
                continue
            try:
                resource.stop() if name == "playwright" else resource.close()
            except Exception as exc:  # pragma: no cover - гасим шум при выходе
                logger.debug("Ошибка при закрытии %s: %s", name, exc)
        self._playwright = self._browser = self._context = self._page = None

    def __enter__(self) -> "AdminUploader":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- вспомогательное ----------------------------------------------------- #

    def _check_authenticated(self) -> None:
        """Убедиться, что сессия жива и нас не выкинуло на логин.

        :raises AuthStateError: похоже, что открыта страница логина.
        """
        current = (self._page.url or "").lower()
        if any(marker in current for marker in LOGIN_URL_MARKERS):
            raise AuthStateError(
                f"Похоже, сессия истекла — админка увела на {self._page.url}. "
                "Обновите её: python auth_state.py"
            )

    def _target(self, name: str, selector: str) -> Any:
        """Вернуть locator поля с учётом того, что оно может жить в iframe."""
        frame_selector = self.selectors.editor_frames.get(name)
        if frame_selector:
            return self._page.frame_locator(frame_selector).locator(selector)
        return self._page.locator(selector)

    def _fill_field(self, name: str, selector: str, value: str) -> None:
        """Заполнить одно поле формы, дождавшись его появления.

        Обычные ``input``/``textarea`` заполняются через ``fill()``. Визуальные
        редакторы (перечисленные в ``rich_text_fields`` или живущие в iframe)
        не поддерживают ``fill()``, поэтому туда содержимое вводится как в
        ``contenteditable``.

        :raises UploaderError: поле не появилось за отведённое время.
        """
        from playwright.sync_api import Error as PlaywrightError

        if not selector:
            logger.debug("Поле '%s' не настроено — пропускаю", name)
            return

        locator = self._target(name, selector)
        try:
            # явное ожидание: элемент должен не просто существовать в DOM,
            # а быть видимым — иначе fill() уйдёт в скрытый черновик формы
            locator.first.wait_for(state="visible", timeout=self.timeout_ms)
        except PlaywrightError as exc:
            raise UploaderError(
                f"Поле '{name}' ({selector}) не появилось: {exc}"
            ) from exc

        is_rich = name in self.selectors.rich_text_fields or name in self.selectors.editor_frames
        try:
            if is_rich:
                locator.first.click()
                locator.first.evaluate("node => { node.innerHTML = ''; }")
                self._page.keyboard.insert_text(value)
            else:
                locator.first.fill(value)
        except PlaywrightError as exc:
            raise UploaderError(f"Не удалось заполнить поле '{name}': {exc}") from exc

        logger.debug("Поле '%s' заполнено (%s символов)", name, len(value))

    def inspect_form(self) -> list[dict[str, Any]]:
        """Перечислить поля формы на странице создания задачи.

        Нужно, чтобы не подбирать селекторы в DevTools вручную: метод открывает
        страницу под сохранённой сессией и возвращает все поля ввода — включая
        те, что живут внутри iframe визуальных редакторов.

        Для каждого поля возвращается готовый CSS-селектор (по ``id``, иначе по
        ``name``, иначе по порядковому номеру), тип элемента, подпись и признак
        ``rich`` — его нужно перечислить в ``rich_text_fields``.

        :raises AuthStateError: сессия истекла.
        :raises UploaderError: страница не открылась.
        """
        from playwright.sync_api import Error as PlaywrightError

        self.open()
        try:
            self._page.goto(self.create_url, wait_until="domcontentloaded")
            self._page.wait_for_load_state("networkidle")
        except PlaywrightError as exc:
            raise UploaderError(f"Не удалось открыть {self.create_url}: {exc}") from exc

        self._check_authenticated()

        fields: list[dict[str, Any]] = []
        for frame in self._page.frames:
            try:
                found = frame.evaluate(_INSPECT_JS)
            except PlaywrightError as exc:  # pragma: no cover - фрейм мог отвалиться
                logger.debug("Фрейм %s не опрошен: %s", frame.url, exc)
                continue

            in_iframe = frame != self._page.main_frame
            for item in found:
                item["frame_url"] = frame.url if in_iframe else ""
                item["in_iframe"] = in_iframe
                fields.append(item)
        return fields

    def _fill_select(self, selector: str, value: str) -> None:
        """Выставить значение выпадающего списка.

        Сначала пробуем найти пункт по видимому тексту («Информатика»), затем по
        его ``value`` — в разметке админок встречается и то, и другое.

        :raises UploaderError: списка нет или в нём нет такого пункта.
        """
        from playwright.sync_api import Error as PlaywrightError

        locator = self._page.locator(selector).first
        try:
            locator.wait_for(state="visible", timeout=self.timeout_ms)
        except PlaywrightError as exc:
            raise UploaderError(f"Список {selector} не появился: {exc}") from exc

        try:
            locator.select_option(label=value)
        except PlaywrightError:
            try:
                locator.select_option(value=value)
            except PlaywrightError as exc:
                raise UploaderError(
                    f"В списке {selector} нет варианта «{value}»: {exc}"
                ) from exc
        logger.debug("Список %s = %s", selector, value)

    def _collect_form_error(self) -> str:
        """Прочитать сообщение об ошибке формы, если админка его показала."""
        selector = self.selectors.error_indicator
        if not selector:
            return ""
        try:
            node = self._page.query_selector(selector)
        except Exception:  # pragma: no cover - зависит от состояния страницы
            return ""
        if node is None:
            return ""
        return (node.inner_text() or "").strip()

    def _screenshot(self, task_id: str) -> Path | None:
        """Сохранить скриншот страницы — чтобы понять, что пошло не так."""
        if self.screenshot_dir is None or self._page is None:
            return None
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = self.screenshot_dir / f"task_{task_id or 'unknown'}_{stamp}.png"
        try:
            self._page.screenshot(path=str(path), full_page=True)
        except Exception as exc:  # pragma: no cover
            logger.debug("Не удалось сохранить скриншот: %s", exc)
            return None
        return path

    # -- публикация ---------------------------------------------------------- #

    def _fill_form(self, solution: TaskSolution) -> None:
        """Разложить данные решения по полям формы.

        Выпадающие списки выставляются первыми: от них в админках обычно зависит
        состав остальной формы (например, поле ответа появляется только после
        выбора типа структуры).

        Текст перед вставкой очищается от Markdown и LaTeX: в поле редактора
        это не разметка, и ученик увидел бы доллары и обратные кавычки
        буквально. Отключается флагом ``render``.
        """
        for selector, value in self.selectors.selects.items():
            self._fill_select(selector, value)

        if self.render:
            condition = render_plain(solution.condition)
            solution_text = render_plain(solution.solution_text)
            hints = render_hints(solution.hints, separator=self.hints_separator)
        else:
            condition = solution.condition
            solution_text = solution.solution_text
            hints = self.hints_separator.join(solution.hints)

        solution = replace(
            solution, condition=condition, solution_text=solution_text
        )
        for name, value in (
            ("task_id", solution.task_id),
            ("condition", solution.condition),
            ("solution", solution.solution_text),
            ("answer", solution.answer),
            ("hints", hints),
        ):
            self._fill_field(name, getattr(self.selectors, name), value)

    def _submit(self, task_id: str) -> UploadResult:
        """Нажать сохранение и дождаться подтверждения.

        :returns: результат публикации; исключение наружу не выпускается — при
            пакетной заливке одна плохая задача не должна ронять весь прогон.
        """
        from playwright.sync_api import Error as PlaywrightError

        selectors = self.selectors
        try:
            button = self._page.locator(selectors.save_button).first
            button.wait_for(state="visible", timeout=self.timeout_ms)
            button.click()
        except PlaywrightError as exc:
            return UploadResult(
                task_id, False, f"Не удалось нажать сохранение: {exc}",
                self._screenshot(task_id),
            )

        error = self._collect_form_error()
        if error:
            return UploadResult(
                task_id, False, f"Админка вернула ошибку: {error}",
                self._screenshot(task_id),
            )

        if selectors.success_indicator:
            try:
                self._page.wait_for_selector(
                    selectors.success_indicator, timeout=self.timeout_ms
                )
            except PlaywrightError as exc:
                return UploadResult(
                    task_id,
                    False,
                    f"Не дождался подтверждения сохранения "
                    f"('{selectors.success_indicator}'): {exc}",
                    self._screenshot(task_id),
                )
        else:
            # Явного индикатора нет — довольствуемся тем, что страница устоялась.
            self._page.wait_for_load_state("networkidle")

        return UploadResult(task_id, True, "опубликовано")

    def publish(self, solution: TaskSolution) -> UploadResult:
        """Опубликовать одну задачу.

        :raises AuthStateError: сессия истекла — дальнейшие попытки бессмысленны.
        """
        from playwright.sync_api import Error as PlaywrightError

        self.open()
        task_id = solution.task_id or "<без id>"
        logger.info("Публикую задачу %s", task_id)

        try:
            self._page.goto(self.create_url, wait_until="domcontentloaded")
        except PlaywrightError as exc:
            return UploadResult(
                task_id, False, f"Не удалось открыть {self.create_url}: {exc}"
            )

        # Протухшая сессия — не повод продолжать: это ошибка окружения, а не задачи.
        self._check_authenticated()

        try:
            self._fill_form(solution)
        except UploaderError as exc:
            return UploadResult(task_id, False, str(exc), self._screenshot(task_id))

        if self.dry_run:
            logger.info("Холостой прогон: форма заполнена, сохранение не нажимаю")
            return UploadResult(task_id, True, "dry-run: форма заполнена")

        result = self._submit(task_id)
        logger.info(
            "Задача %s: %s", task_id, "успех" if result.ok else f"ошибка — {result.message}"
        )
        return result

    def publish_many(self, solutions: Iterable[TaskSolution]) -> list[UploadResult]:
        """Опубликовать несколько задач подряд, переиспользуя одну вкладку.

        Ошибка одной задачи не прерывает остальные; исключение пробрасывается
        только при протухшей сессии — тогда продолжать смысла нет.
        """
        results: list[UploadResult] = []
        for solution in solutions:
            results.append(self.publish(solution))
        return results


async def publish_async(uploader: AdminUploader, solution: TaskSolution) -> UploadResult:
    """Асинхронная обёртка над :meth:`AdminUploader.publish`.

    Загрузчик синхронный и держит одну вкладку, поэтому обращения к одному
    экземпляру нельзя распараллеливать — обёртка нужна лишь для того, чтобы не
    блокировать событийный цикл ``main.py`` во время публикации.
    """
    import asyncio

    return await asyncio.to_thread(uploader.publish, solution)


# --------------------------------------------------------------------------- #
# CLI — отладка селекторов
# --------------------------------------------------------------------------- #


def load_solution(path: Path) -> TaskSolution:
    """Прочитать :class:`TaskSolution` из JSON-файла.

    :raises UploaderError: файл нечитаем или в нём нет нужных полей.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise UploaderError(f"Не удалось прочитать {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise UploaderError(f"{path}: ожидался JSON-объект")

    missing = [
        key
        for key in ("task_id", "condition", "solution_text", "answer")
        if key not in data
    ]
    if missing:
        raise UploaderError(f"{path}: отсутствуют поля {', '.join(missing)}")

    hints = data.get("hints") or []
    if isinstance(hints, str):
        hints = [hints]

    return TaskSolution(
        task_id=str(data["task_id"]),
        condition=str(data["condition"]),
        solution_text=str(data["solution_text"]),
        answer=str(data["answer"]),
        hints=[str(hint) for hint in hints],
    )


def print_form_fields(fields: list[dict[str, Any]], *, show_all: bool = False) -> None:
    """Напечатать найденные поля формы — чтобы скопировать селекторы в конфиг.

    :param show_all: показывать и кнопки панелей визуальных редакторов. По
        умолчанию они скрыты: одна панель Froala добавляет полсотни кнопок
        «Bold», «Italic» и прочих, среди которых полезные поля теряются.
    """
    hidden = 0
    if not show_all:
        total = len(fields)
        fields = [item for item in fields if not item.get("chrome")]
        hidden = total - len(fields)

    if not fields:
        print("Полей ввода на странице не найдено. Возможно, форма грузится позже "
              "или лежит в iframe, недоступном для опроса.")
        return

    print(f"\nНайдено полей: {len(fields)}")
    if hidden:
        print(f"(скрыто кнопок панелей редакторов: {hidden}; показать — с --all)")
    print("=" * 78)
    for item in fields:
        kind = item["tag"]
        if item["type"]:
            kind += f"[{item['type']}]"
        if item["rich"]:
            kind += " (визуальный редактор)"
        if item["in_iframe"]:
            kind += " (внутри iframe)"

        print(f"\n  селектор : {item['selector']}")
        print(f"  тип      : {kind}")
        for key, title in (
            ("label", "подпись "),
            ("placeholder", "плейсхолдер"),
            ("name", "name    "),
            ("text", "текст   "),
        ):
            if item.get(key):
                print(f"  {title} : {item[key]}")
        if not item["visible"]:
            print("  ВНИМАНИЕ : элемент сейчас невидим — заполнение будет ждать его")

    rich = [item["selector"] for item in fields if item["rich"] or item["in_iframe"]]
    print("\n" + "=" * 78)
    print("Перенесите нужные селекторы в FieldSelectors (uploader.py).")
    if rich:
        print(f"Поля {', '.join(rich)} — визуальные редакторы: перечислите их имена "
              "в rich_text_fields, а для iframe заполните editor_frames.")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Разобрать аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description="Публикация готовой задачи в админку сайта."
    )
    parser.add_argument(
        "solution",
        type=Path,
        nargs="?",
        help="JSON с решением (результат llm_processor); не нужен при --inspect",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Показать поля формы создания задачи и выйти (подбор селекторов)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="При --inspect показывать и кнопки панелей визуальных редакторов",
    )
    parser.add_argument(
        "--create-url",
        default=DEFAULT_CREATE_URL,
        help="Страница создания задачи (по умолчанию ADMIN_CREATE_TASK_URL из .env)",
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=DEFAULT_STATE_PATH,
        help=f"Файл сессии (по умолчанию {DEFAULT_STATE_PATH})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Заполнить форму, но не нажимать сохранение",
    )
    parser.add_argument(
        "--headed", action="store_true", help="Показать окно браузера (отладка)"
    )
    parser.add_argument(
        "--screenshot-dir",
        type=Path,
        default=None,
        help="Куда сохранять скриншоты неудачных публикаций",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Точка входа CLI. Возвращает код возврата процесса."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = _parse_args(argv)

    if not args.inspect and args.solution is None:
        logger.error("Укажите JSON с решением или запустите с --inspect")
        return 1

    try:
        solution = None if args.inspect else load_solution(args.solution)
        uploader = AdminUploader(
            create_url=args.create_url,
            state_path=args.state,
            headless=not args.headed,
            dry_run=args.dry_run,
            screenshot_dir=args.screenshot_dir,
        )
    except UploaderError as exc:
        logger.error("%s", exc)
        return 1

    with uploader:
        try:
            if args.inspect:
                print_form_fields(uploader.inspect_form(), show_all=args.all)
                return 0
            result = uploader.publish(solution)
        except UploaderError as exc:
            logger.error("%s", exc)
            return 1

    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
