"""Выкачивание базы заданий из админки умскул через её JSON API.

Эндпоинт ``/api/materials/tasks/`` отдаёт задачу целиком: условие, решение,
подсказки, ответы, тему, сложность и служебные флаги. Он же умеет фильтровать и
листать, поэтому базу можно снять прототип за прототипом::

    /api/materials/tasks/?course_type_task_number=943&limit=100

Зачем это нужно: чтобы отбирать актуальные задания, их надо сначала увидеть все
разом, а не ходить по двум тысячам вкладок. Модуль складывает базу в локальный
JSONL, дальше с ней работают инструменты отбора — без сети и без лимитов.

Доступ закрыт логином, поэтому используются куки из ``auth_state.json``, который
создаёт ``auth_state.py``. Отдельный вход не нужен.

Примеры::

    python umschool.py --task 323750                 # одна задача, сырой JSON
    python umschool.py --prototype 943 --out db.jsonl
    python umschool.py --param class_type=11 --param class_year=3 --out db.jsonl
    python umschool.py --stats db.jsonl              # что скачалось
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

#: База API админки.
DEFAULT_API_URL = os.getenv("UMSCHOOL_API_URL", "https://old.umschool.net/api/materials")

#: Файл сессии, созданный auth_state.py.
DEFAULT_STATE_PATH = Path(os.getenv("AUTH_STATE_PATH", "auth_state.json"))

#: Сколько задач запрашивать за раз.
DEFAULT_PAGE_SIZE = 100

#: Куда складывать выкачанную базу.
DEFAULT_DB_PATH = Path("output/umschool_tasks.jsonl")

#: Признаки того, что сервер увёл нас на логин вместо данных.
LOGIN_MARKERS = ("<html", "login", "signin")


class UmschoolError(Exception):
    """Не удалось получить данные из админки."""


class UmschoolAuthError(UmschoolError):
    """Сессия отсутствует или истекла — нужен повторный ``auth_state.py``."""


# --------------------------------------------------------------------------- #
# Модель данных
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class UmTask:
    """Задача из базы умскул — только то, что нужно для отбора.

    Сырой JSON сохраняется целиком в :attr:`raw`, поэтому ничего не теряется:
    если позже понадобится поле, о котором сейчас не думали, перечитывать базу
    заново не придётся.
    """

    id: int
    prototype_id: int | None = None
    prototype_code: str = ""
    prototype_title: str = ""
    prototype_order: int | None = None
    topic_id: int | None = None
    topic_code: str = ""
    topic_title: str = ""
    condition_html: str = ""
    solution_html: str = ""
    extra_text: str = ""
    answers: list[str] = field(default_factory=list)
    tips: list[str] = field(default_factory=list)
    images: list[str] = field(default_factory=list)
    difficulty: str = ""
    max_points: int = 0
    type_display: str = ""
    structure_type_display: str = ""
    for_generation: bool = False
    checked_by_expert: bool = False
    is_previously_used: bool = False
    created_at: str = ""
    updated_at: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def url(self) -> str:
        """Ссылка на задачу в админке — по ней удобно открыть и поправить."""
        return f"https://old.umschool.net/materials/tasks/{self.id}"

    def to_dict(self) -> dict[str, Any]:
        """Словарь для записи в JSONL."""
        return asdict(self)


#: Ссылки на картинки внутри условия: в задании 1 вся суть именно в них.
IMAGE_PATTERN = re.compile(r'<img[^>]+src="([^"]+)"', re.IGNORECASE)


def parse_task(payload: dict[str, Any]) -> UmTask:
    """Разобрать задачу из ответа API.

    Вложенные объекты (тема, прототип, сложность) распаковываются в плоские
    поля: так с базой удобнее работать при отборе и в таблицах.
    """
    prototype = payload.get("course_type_task_number") or {}
    topic = payload.get("class_type_topic") or {}
    difficulty = payload.get("difficulty") or {}

    condition = payload.get("title") or ""
    return UmTask(
        id=int(payload["id"]),
        prototype_id=prototype.get("id"),
        prototype_code=prototype.get("code") or "",
        prototype_title=prototype.get("title") or "",
        prototype_order=prototype.get("order"),
        topic_id=topic.get("id"),
        topic_code=topic.get("code") or "",
        topic_title=topic.get("title") or "",
        condition_html=condition,
        solution_html=payload.get("solution") or "",
        extra_text=payload.get("extra_text") or "",
        answers=[
            str(variant.get("variant", "")).strip()
            for variant in payload.get("variants") or []
            if variant.get("is_correct")
        ],
        tips=[
            str(tip.get("tip", "")).strip()
            for tip in payload.get("tips") or []
            if tip.get("tip")
        ],
        images=IMAGE_PATTERN.findall(condition),
        difficulty=difficulty.get("title") or "",
        max_points=payload.get("max_points") or 0,
        type_display=payload.get("type_display") or "",
        structure_type_display=payload.get("structure_type_display") or "",
        for_generation=bool(payload.get("for_generation")),
        checked_by_expert=bool(payload.get("checked_by_expert")),
        is_previously_used=bool(payload.get("is_previously_used")),
        created_at=str(payload.get("created_at") or ""),
        updated_at=str(payload.get("updated_at") or ""),
        raw=payload,
    )


# --------------------------------------------------------------------------- #
# Сессия
# --------------------------------------------------------------------------- #


def load_cookies(state_path: Path = DEFAULT_STATE_PATH) -> dict[str, str]:
    """Достать куки из ``auth_state.json``.

    Файл создаёт ``auth_state.py`` для Playwright, но формат простой, и те же
    куки годятся для обычных HTTP-запросов — второй раз логиниться незачем.

    :raises UmschoolAuthError: файла нет или в нём нет кук.
    """
    state_path = Path(state_path)
    if not state_path.exists():
        raise UmschoolAuthError(
            f"Файл сессии {state_path} не найден. Выполните: python auth_state.py"
        )

    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise UmschoolAuthError(f"Не удалось прочитать {state_path}: {exc}") from exc

    cookies = {
        str(cookie["name"]): str(cookie.get("value", ""))
        for cookie in data.get("cookies", [])
        if cookie.get("name")
    }
    if not cookies:
        raise UmschoolAuthError(f"В {state_path} нет кук — обновите сессию")
    return cookies


# --------------------------------------------------------------------------- #
# Клиент
# --------------------------------------------------------------------------- #


class UmschoolClient:
    """Читает задачи из API админки.

    :param api_url: база API.
    :param state_path: файл сессии.
    :param page_size: сколько задач запрашивать за раз.
    :param request_delay: пауза между запросами, сек — вежливость к серверу.
    :param timeout: таймаут запроса, сек.
    """

    def __init__(
        self,
        *,
        api_url: str = DEFAULT_API_URL,
        state_path: Path = DEFAULT_STATE_PATH,
        page_size: int = DEFAULT_PAGE_SIZE,
        request_delay: float = 0.3,
        timeout: float = 30.0,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.page_size = page_size
        self.request_delay = request_delay
        self.timeout = timeout

        self.session = requests.Session()
        self.session.cookies.update(load_cookies(state_path))
        self.session.headers.update(
            {"Accept": "application/json", "X-Requested-With": "XMLHttpRequest"}
        )
        self._last_request_at = 0.0

    def close(self) -> None:
        """Закрыть HTTP-сессию."""
        self.session.close()

    def __enter__(self) -> "UmschoolClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _throttle(self) -> None:
        """Выдержать паузу между запросами."""
        elapsed = time.monotonic() - self._last_request_at
        if self._last_request_at and elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self._last_request_at = time.monotonic()

    def _get(self, url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Выполнить запрос и разобрать JSON.

        :raises UmschoolAuthError: вместо данных пришла страница логина.
        :raises UmschoolError: сетевая ошибка или не-JSON в ответе.
        """
        self._throttle()
        logger.debug("GET %s params=%s", url, params)
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
        except requests.RequestException as exc:
            raise UmschoolError(f"Запрос {url} не удался: {exc}") from exc

        if response.status_code in {401, 403}:
            raise UmschoolAuthError(
                f"Админка отвечает {response.status_code} — сессия истекла. "
                "Обновите её: python auth_state.py"
            )
        if response.status_code >= 400:
            raise UmschoolError(f"{url} вернул {response.status_code}")

        body = response.text.lstrip()
        if body[:1] not in "{[":
            # HTML вместо JSON почти всегда означает редирект на форму входа.
            hint = "сессия истекла" if any(m in body[:400].lower() for m in LOGIN_MARKERS) else "не JSON"
            raise UmschoolAuthError(f"{url} вернул не данные ({hint})")

        try:
            return response.json()
        except requests.JSONDecodeError as exc:
            raise UmschoolError(f"{url} вернул неразбираемый JSON: {exc}") from exc

    # -- публичный API ------------------------------------------------------- #

    def fetch_task(self, task_id: int | str) -> UmTask:
        """Получить одну задачу по её идентификатору.

        :raises UmschoolError: задача не найдена.
        """
        payload = self._get(f"{self.api_url}/tasks/", {"id": task_id})
        results = payload.get("results") or []
        if not results:
            raise UmschoolError(f"Задача {task_id} не найдена")
        return parse_task(results[0])

    def iter_tasks(self, **filters: Any) -> Iterator[UmTask]:
        """Пройти по всем задачам, подходящим под фильтр.

        Постранично, по ссылке ``next`` из ответа — так не нужно гадать про
        смещения и признак конца.

        :param filters: параметры запроса, например
            ``course_type_task_number=943`` или ``class_type=11``.
        """
        params: dict[str, Any] | None = {**filters, "limit": self.page_size}
        url = f"{self.api_url}/tasks/"
        seen = 0

        while url:
            payload = self._get(url, params)
            params = None  # в ссылке next параметры уже зашиты

            results = payload.get("results") or []
            for item in results:
                seen += 1
                yield parse_task(item)

            total = payload.get("count")
            if total is not None and seen:
                logger.info("Скачано %s из %s", seen, total)
            url = payload.get("next") or ""

    def count_tasks(self, **filters: Any) -> int:
        """Узнать, сколько задач подходит под фильтр, не выкачивая их."""
        payload = self._get(f"{self.api_url}/tasks/", {**filters, "limit": 1})
        return int(payload.get("count") or 0)


# --------------------------------------------------------------------------- #
# Локальная база
# --------------------------------------------------------------------------- #


def save_tasks(tasks: Iterator[UmTask], path: Path) -> int:
    """Записать задачи в JSONL, по одной на строку.

    Формат выбран нарочно: файл дописывается по мере выкачки, поэтому обрыв на
    середине не теряет уже скачанное, а читать его можно построчно, не поднимая
    всю базу в память.

    :returns: сколько задач записано.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count


def load_tasks(path: Path) -> list[UmTask]:
    """Прочитать локальную базу из JSONL."""
    tasks: list[UmTask] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                tasks.append(UmTask(**json.loads(line)))
    return tasks


def print_stats(tasks: Sequence[UmTask]) -> None:
    """Показать, что лежит в базе: прототипы, темы, сложность.

    Это первый взгляд на масштаб отбора — сколько задач на каждую тему и где их
    заведомо больше, чем нужно.
    """
    if not tasks:
        print("База пуста.")
        return

    print(f"\nВсего задач: {len(tasks)}")

    by_prototype: Counter[str] = Counter()
    for task in tasks:
        name = f"{task.prototype_order or '?'}. {task.prototype_title or 'без прототипа'}"
        by_prototype[name] += 1

    for prototype, total in sorted(by_prototype.items()):
        print(f"\n{prototype} — {total} задач")
        topics: Counter[str] = Counter()
        for task in tasks:
            label = f"{task.prototype_order or '?'}. {task.prototype_title or 'без прототипа'}"
            if label == prototype:
                topics[task.topic_title or "тема не проставлена"] += 1
        for topic, count in sorted(topics.items()):
            print(f"    {count:>4}  {topic}")

    without_topic = sum(1 for task in tasks if not task.topic_title)
    with_images = sum(1 for task in tasks if task.images)
    checked = sum(1 for task in tasks if task.checked_by_expert)
    print(
        f"\nБез темы: {without_topic}   с картинками: {with_images}   "
        f"проверено экспертом: {checked}"
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_params(pairs: Sequence[str]) -> dict[str, str]:
    """Разобрать аргументы ``--param ключ=значение``.

    :raises UmschoolError: аргумент записан без «=».
    """
    params: dict[str, str] = {}
    for pair in pairs:
        key, separator, value = pair.partition("=")
        if not separator or not key.strip():
            raise UmschoolError(f"Ожидался формат ключ=значение, получено: {pair!r}")
        params[key.strip()] = value.strip()
    return params


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Разобрать аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description="Выкачивание базы заданий умскул через API админки."
    )
    parser.add_argument("--task", help="Скачать одну задачу по id и показать её")
    parser.add_argument(
        "--prototype",
        help="Скачать все задачи прототипа (course_type_task_number), например 943",
    )
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="КЛЮЧ=ЗНАЧЕНИЕ",
        help="Произвольный фильтр запроса; можно указывать несколько раз",
    )
    parser.add_argument(
        "--count",
        action="store_true",
        help="Только узнать количество задач под фильтр, не выкачивая",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"Куда сохранить базу (по умолчанию {DEFAULT_DB_PATH})",
    )
    parser.add_argument("--stats", type=Path, help="Показать сводку по готовой базе")
    parser.add_argument(
        "--state",
        type=Path,
        default=DEFAULT_STATE_PATH,
        help=f"Файл сессии (по умолчанию {DEFAULT_STATE_PATH})",
    )
    parser.add_argument(
        "--api-url", default=DEFAULT_API_URL, help=f"База API (по умолчанию {DEFAULT_API_URL})"
    )
    parser.add_argument("--verbose", action="store_true", help="Подробный лог")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Точка входа CLI. Возвращает код возврата процесса."""
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.stats:
        print_stats(load_tasks(args.stats))
        return 0

    try:
        filters = parse_params(args.param)
        if args.prototype:
            filters["course_type_task_number"] = args.prototype

        client = UmschoolClient(api_url=args.api_url, state_path=args.state)
    except UmschoolError as exc:
        logger.error("%s", exc)
        return 1

    with client:
        try:
            if args.task:
                task = client.fetch_task(args.task)
                print(json.dumps(task.to_dict(), ensure_ascii=False, indent=2))
                return 0

            if not filters:
                logger.error(
                    "Задайте, что выкачивать: --prototype, --task или --param"
                )
                return 1

            if args.count:
                print(f"Задач под фильтр: {client.count_tasks(**filters)}")
                return 0

            logger.info("Выкачиваю задачи: %s", filters)
            total = save_tasks(client.iter_tasks(**filters), args.out)
        except UmschoolError as exc:
            logger.error("%s", exc)
            return 1

    logger.info("Сохранено задач: %s → %s", total, args.out)
    print_stats(load_tasks(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
