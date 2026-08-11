"""Сохранение сессии админки для повторного использования в ``uploader.py``.

Скрипт открывает *видимый* браузер, ждёт, пока вы вручную залогинитесь
(включая капчу, 2FA и любые другие шаги), и записывает состояние контекста
(куки + localStorage) в ``auth_state.json``. Дальше ``uploader.py`` поднимает
контекст из этого файла и работает уже авторизованным.

Здесь намеренно используется **синхронный** Playwright: сценарий интерактивный,
выполняется вручную и раз в несколько дней — асинхронность только усложнила бы
код без выигрыша.

Запуск::

    python auth_state.py                                  # URL из ADMIN_LOGIN_URL
    python auth_state.py --url https://site.ru/admin/login
    python auth_state.py --wait-selector ".admin-sidebar" # ждать элемент вместо Enter

Файл ``auth_state.json`` содержит действующую сессию — он в ``.gitignore``,
не коммитьте его и не передавайте третьим лицам.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import stat
import sys
from pathlib import Path

from dotenv import load_dotenv
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

load_dotenv()

logger = logging.getLogger(__name__)

#: Куда по умолчанию складывается состояние сессии.
DEFAULT_STATE_PATH = Path(os.getenv("AUTH_STATE_PATH", "auth_state.json"))

#: Стартовая страница логина.
DEFAULT_LOGIN_URL = os.getenv("ADMIN_LOGIN_URL", "")

#: Сколько ждать появления --wait-selector, мс.
DEFAULT_WAIT_TIMEOUT_MS = 5 * 60 * 1000


def _restrict_permissions(path: Path) -> None:
    """Оставить доступ к файлу сессии только владельцу (chmod 600).

    На Windows вызов игнорируется — там модель прав другая.
    """
    try:
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except OSError as exc:  # pragma: no cover - зависит от ОС/ФС
        logger.debug("Не удалось выставить права 600 на %s: %s", path, exc)


def _describe_state(path: Path) -> str:
    """Короткая сводка о сохранённом состоянии — для лога."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "не удалось прочитать файл состояния"

    cookies = len(data.get("cookies", []))
    origins = len(data.get("origins", []))
    return f"куки: {cookies}, origins с localStorage: {origins}"


def save_auth_state(
    login_url: str,
    state_path: Path = DEFAULT_STATE_PATH,
    *,
    browser_name: str = "chromium",
    wait_selector: str | None = None,
    wait_timeout_ms: int = DEFAULT_WAIT_TIMEOUT_MS,
) -> Path:
    """Открыть браузер, дождаться ручного логина и сохранить состояние сессии.

    :param login_url: страница логина админки.
    :param state_path: куда записать состояние.
    :param browser_name: ``chromium`` | ``firefox`` | ``webkit``.
    :param wait_selector: CSS-селектор элемента, который появляется только после
        успешного логина. Если задан — скрипт ждёт его вместо нажатия Enter.
    :param wait_timeout_ms: таймаут ожидания селектора.
    :returns: путь к сохранённому файлу состояния.
    :raises ValueError: если не передан URL логина.
    :raises RuntimeError: если логин не подтверждён (селектор не дождались).
    """
    if not login_url:
        raise ValueError(
            "Не указан URL логина: передайте --url или задайте ADMIN_LOGIN_URL в .env"
        )

    state_path = Path(state_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    if state_path.exists():
        logger.warning("Файл %s уже существует и будет перезаписан", state_path)

    with sync_playwright() as playwright:
        browser_type = getattr(playwright, browser_name)
        # headless=False обязателен: логин выполняется руками.
        browser = browser_type.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()

        try:
            logger.info("Открываю %s", login_url)
            page.goto(login_url, wait_until="domcontentloaded")

            if wait_selector:
                print(
                    f"\n>>> Войдите в админку в открытом окне браузера.\n"
                    f">>> Жду появления элемента '{wait_selector}' "
                    f"(до {wait_timeout_ms // 1000} сек)...\n"
                )
                try:
                    page.wait_for_selector(wait_selector, timeout=wait_timeout_ms)
                except PlaywrightError as exc:
                    raise RuntimeError(
                        f"Элемент '{wait_selector}' не появился — логин не подтверждён"
                    ) from exc
            else:
                if not sys.stdin.isatty():
                    raise RuntimeError(
                        "Нет интерактивного терминала для ожидания Enter. "
                        "Запустите скрипт вручную или используйте --wait-selector."
                    )
                print(
                    "\n>>> Войдите в админку в открытом окне браузера.\n"
                    ">>> Когда окажетесь внутри — вернитесь сюда и нажмите Enter.\n"
                )
                input(">>> Нажмите Enter для сохранения сессии... ")

            context.storage_state(path=str(state_path))
        finally:
            context.close()
            browser.close()

    _restrict_permissions(state_path)
    logger.info("Сессия сохранена в %s (%s)", state_path, _describe_state(state_path))
    return state_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Разобрать аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description="Ручной логин в админку и сохранение сессии Playwright."
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_LOGIN_URL,
        help="URL страницы логина (по умолчанию — ADMIN_LOGIN_URL из .env)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_STATE_PATH,
        help=f"Куда сохранить состояние (по умолчанию {DEFAULT_STATE_PATH})",
    )
    parser.add_argument(
        "--browser",
        choices=("chromium", "firefox", "webkit"),
        default="chromium",
        help="Движок браузера (по умолчанию chromium)",
    )
    parser.add_argument(
        "--wait-selector",
        default=None,
        help="CSS-селектор, появляющийся после успешного логина; "
        "если задан — ждём его вместо нажатия Enter",
    )
    parser.add_argument(
        "--wait-timeout",
        type=int,
        default=DEFAULT_WAIT_TIMEOUT_MS,
        help="Таймаут ожидания селектора в миллисекундах",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Точка входа CLI. Возвращает код возврата процесса."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = _parse_args(argv)

    try:
        save_auth_state(
            args.url,
            args.output,
            browser_name=args.browser,
            wait_selector=args.wait_selector,
            wait_timeout_ms=args.wait_timeout,
        )
    except (ValueError, RuntimeError, PlaywrightError) as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
