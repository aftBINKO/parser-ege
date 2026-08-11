"""Интеграция с Gemini API: превращает сырой текст задачи в структурированный JSON.

Модуль отвечает ровно за один шаг пайплайна: получить на вход текст условия
(как его вернул ``scraper.py``) и вернуть провалидированный объект
:class:`TaskSolution` с полями ``task_id``, ``condition``, ``solution_text``,
``answer``, ``hints``.

Ключевые решения:

* Модель работает в режиме ``response_mime_type="application/json"`` — Gemini
  сам гарантирует синтаксически валидный JSON, но мы всё равно не доверяем
  ответу и парсим его защитно (см. :func:`parse_llm_json`).
* Все сетевые вызовы асинхронные (``generate_content_async``), поэтому пачку
  задач можно обрабатывать конкурентно через :meth:`GeminiProcessor.process_many`.
* Ошибки разделены на два класса: проблема с сетью/API (:class:`LLMRequestError`)
  и проблема с содержимым ответа (:class:`LLMResponseError`). Оркестратор может
  реагировать на них по-разному.

Пример использования::

    import asyncio
    from llm_processor import GeminiProcessor

    async def main() -> None:
        processor = GeminiProcessor()
        result = await processor.process(raw_text, task_id="12345")
        print(result.answer)

    asyncio.run(main())
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Константы
# --------------------------------------------------------------------------- #

#: Модель по умолчанию; переопределяется переменной окружения ``GEMINI_MODEL``.
DEFAULT_MODEL = "gemini-2.5-flash"

#: Поля, которые обязаны присутствовать в ответе модели.
REQUIRED_FIELDS: tuple[str, ...] = (
    "task_id",
    "condition",
    "solution_text",
    "answer",
    "hints",
)

#: Жёсткая системная инструкция. Меняется только осознанно: от неё напрямую
#: зависит стабильность парсинга ответа.
SYSTEM_INSTRUCTION = """\
Ты — методист по информатике, который готовит разборы задач ЕГЭ для базы знаний
образовательного сайта.

На вход ты получаешь сырой текст задачи (возможно, с артефактами вёрстки:
лишними переносами строк, ссылками на изображения, пробелами).

ТВОЯ ЗАДАЧА:
1. Очистить и восстановить условие задачи.
2. Решить задачу.
3. Написать пошаговое решение, понятное школьнику.
4. Дать финальный ответ.
5. Сформулировать подсказки, которые ведут к решению, но НЕ раскрывают ответ.

ФОРМАТ ОТВЕТА — СТРОГО ОДИН JSON-ОБЪЕКТ, БЕЗ ЛЮБОГО ТЕКСТА ДО И ПОСЛЕ,
БЕЗ MARKDOWN-ОГРАЖДЕНИЙ (```), со следующими полями:

{
  "task_id": "строка — идентификатор задачи, если он есть в тексте, иначе пустая строка",
  "condition": "строка — очищенное условие задачи",
  "solution_text": "строка — подробное пошаговое решение",
  "answer": "строка — только финальный ответ, без пояснений",
  "hints": ["строка", "строка", "..."]
}

ЖЁСТКИЕ ПРАВИЛА:
- Все пять полей обязательны. Ни одно поле нельзя пропустить.
- "hints" — всегда массив строк (от 1 до 3 элементов). Если подсказок нет — [].
- "answer" — всегда строка, даже если ответ числовой (например: "42").
- Формулы оформляй в LaTeX внутри $...$.
- Никогда не выдумывай данные, которых нет в условии. Если условие
  нечитаемо или неполно, верни поле "answer" со значением "" и опиши проблему
  в "solution_text".
- Не добавляй в JSON никаких других полей.
"""


# --------------------------------------------------------------------------- #
# Исключения
# --------------------------------------------------------------------------- #


class LLMError(Exception):
    """Базовая ошибка модуля."""


class LLMRequestError(LLMError):
    """Не удалось получить ответ от API (сеть, лимиты, блокировка контента)."""


class LLMResponseError(LLMError):
    """Ответ получен, но его не удалось распарсить или он не прошёл валидацию."""

    def __init__(self, message: str, raw_response: str = "") -> None:
        super().__init__(message)
        self.raw_response = raw_response


# --------------------------------------------------------------------------- #
# Модель данных
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class TaskSolution:
    """Результат обработки одной задачи — контракт между LLM и ``uploader.py``."""

    task_id: str
    condition: str
    solution_text: str
    answer: str
    hints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Вернуть словарь, пригодный для сериализации в JSON."""
        return asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        """Вернуть JSON-строку (UTF-8, без экранирования кириллицы)."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


# --------------------------------------------------------------------------- #
# Парсинг ответа модели
# --------------------------------------------------------------------------- #


def _strip_code_fences(raw: str) -> str:
    """Убрать markdown-ограждения ```json ... ``` вокруг ответа, если они есть."""
    text = raw.strip()
    if not text.startswith("```"):
        return text

    lines = text.splitlines()
    lines = lines[1:]  # строка вида ``` или ```json
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _extract_json_object(text: str) -> str:
    """Выделить первый сбалансированный JSON-объект из текста.

    Нужно на случай, когда модель всё же добавила пояснение до или после JSON.
    Учитывает строковые литералы и экранирование, поэтому фигурная скобка
    внутри строки не ломает подсчёт вложенности.

    :raises LLMResponseError: если объект не найден или скобки не сбалансированы.
    """
    start = text.find("{")
    if start == -1:
        raise LLMResponseError("В ответе модели нет JSON-объекта", text)

    depth = 0
    in_string = False
    escaped = False

    for index in range(start, len(text)):
        char = text[index]

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    raise LLMResponseError("JSON-объект в ответе не закрыт", text)


def _coerce_hints(value: Any) -> list[str]:
    """Привести поле ``hints`` к списку строк.

    Модель иногда возвращает подсказки одной строкой или списком объектов —
    приводим оба случая к плоскому списку строк вместо того, чтобы падать.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, Sequence):
        hints: list[str] = []
        for item in value:
            text = item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
            text = text.strip()
            if text:
                hints.append(text)
        return hints
    raise LLMResponseError(f"Поле 'hints' имеет недопустимый тип: {type(value).__name__}")


def parse_llm_json(raw: str, *, fallback_task_id: str | None = None) -> TaskSolution:
    """Распарсить сырой ответ модели в :class:`TaskSolution`.

    :param raw: текст ответа модели.
    :param fallback_task_id: чем заполнить ``task_id``, если модель его не вернула
        (обычно — идентификатор, известный парсеру из URL).
    :raises LLMResponseError: если ответ не JSON, не объект или в нём нет
        обязательных полей.
    """
    if not raw or not raw.strip():
        raise LLMResponseError("Модель вернула пустой ответ", raw)

    candidate = _extract_json_object(_strip_code_fences(raw))

    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise LLMResponseError(f"Не удалось разобрать JSON: {exc}", raw) from exc

    if not isinstance(data, dict):
        raise LLMResponseError(
            f"Ожидался JSON-объект, получен {type(data).__name__}", raw
        )

    if fallback_task_id and not str(data.get("task_id") or "").strip():
        data["task_id"] = fallback_task_id

    missing = [key for key in REQUIRED_FIELDS if key not in data]
    if missing:
        raise LLMResponseError(
            f"В ответе модели отсутствуют поля: {', '.join(missing)}", raw
        )

    return TaskSolution(
        task_id=str(data["task_id"] or "").strip(),
        condition=str(data["condition"] or "").strip(),
        solution_text=str(data["solution_text"] or "").strip(),
        answer=str(data["answer"] or "").strip(),
        hints=_coerce_hints(data["hints"]),
    )


# --------------------------------------------------------------------------- #
# Клиент Gemini
# --------------------------------------------------------------------------- #


class GeminiProcessor:
    """Асинхронная обёртка над Gemini API с ретраями и валидацией ответа.

    :param api_key: ключ API. По умолчанию берётся из ``GEMINI_API_KEY``.
    :param model_name: имя модели. По умолчанию — из ``GEMINI_MODEL``.
    :param system_instruction: системный промпт; по умолчанию
        :data:`SYSTEM_INSTRUCTION`.
    :param temperature: температура генерации. Для разборов задач нужен
        предсказуемый результат, поэтому значение по умолчанию низкое.
    :param max_retries: сколько раз повторить запрос при сбое.
    :param retry_base_delay: базовая задержка экспоненциального бэкоффа, сек.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model_name: str | None = None,
        system_instruction: str = SYSTEM_INSTRUCTION,
        temperature: float = 0.2,
        max_retries: int = 3,
        retry_base_delay: float = 2.0,
    ) -> None:
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise LLMError(
                "Не задан GEMINI_API_KEY: положите его в .env или передайте в конструктор"
            )

        self.model_name = model_name or os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
        self.max_retries = max(1, max_retries)
        self.retry_base_delay = retry_base_delay

        genai.configure(api_key=key)
        self._model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=system_instruction,
            generation_config={
                "temperature": temperature,
                # Просим API отдавать именно JSON — это снимает большую часть
                # проблем с markdown-ограждениями и болтовнёй вокруг ответа.
                "response_mime_type": "application/json",
            },
        )
        logger.debug("GeminiProcessor инициализирован (модель=%s)", self.model_name)

    # -- внутреннее ---------------------------------------------------------- #

    @staticmethod
    def _build_prompt(raw_text: str, task_id: str | None) -> str:
        """Собрать пользовательскую часть промпта."""
        header = f"ID задачи: {task_id}\n\n" if task_id else ""
        return f"{header}Сырой текст задачи:\n<<<\n{raw_text.strip()}\n>>>"

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Достать текст из ответа SDK, не роняя пайплайн на пустых кандидатах."""
        text = getattr(response, "text", None)
        if text:
            return text

        # Запасной путь: ответ мог быть отфильтрован или разбит на части.
        parts: list[str] = []
        for candidate in getattr(response, "candidates", None) or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", None) or []:
                part_text = getattr(part, "text", None)
                if part_text:
                    parts.append(part_text)
        return "".join(parts)

    async def _generate(self, prompt: str) -> str:
        """Выполнить запрос к API с экспоненциальным бэкоффом.

        :raises LLMRequestError: если все попытки исчерпаны.
        """
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = await self._model.generate_content_async(prompt)
            except Exception as exc:  # SDK бросает разнородные исключения
                last_error = exc
                logger.warning(
                    "Запрос к Gemini не удался (попытка %s/%s): %s",
                    attempt,
                    self.max_retries,
                    exc,
                )
            else:
                text = self._extract_text(response)
                if text.strip():
                    return text
                last_error = LLMRequestError("Модель вернула пустой ответ")
                logger.warning(
                    "Пустой ответ Gemini (попытка %s/%s)", attempt, self.max_retries
                )

            if attempt < self.max_retries:
                # jitter, чтобы параллельные запросы не били по лимитам синхронно
                delay = self.retry_base_delay * 2 ** (attempt - 1)
                await asyncio.sleep(delay + random.uniform(0, 0.5))

        raise LLMRequestError(
            f"Не удалось получить ответ от Gemini за {self.max_retries} попыт(ок): {last_error}"
        ) from last_error

    # -- публичный API ------------------------------------------------------- #

    async def process(self, raw_text: str, *, task_id: str | None = None) -> TaskSolution:
        """Обработать одну задачу.

        :param raw_text: сырой текст условия из ``scraper.py``.
        :param task_id: известный идентификатор задачи; подставится в результат,
            если модель не вернёт свой.
        :raises LLMRequestError: сбой обращения к API.
        :raises LLMResponseError: ответ не удалось распарсить/провалидировать.
        """
        if not raw_text or not raw_text.strip():
            raise LLMResponseError("На вход подан пустой текст задачи")

        logger.info("Отправляю задачу %s в Gemini", task_id or "<без id>")
        raw_response = await self._generate(self._build_prompt(raw_text, task_id))
        solution = parse_llm_json(raw_response, fallback_task_id=task_id)
        logger.info("Задача %s обработана", solution.task_id or "<без id>")
        return solution

    async def process_many(
        self,
        items: Iterable[tuple[str | None, str]],
        *,
        concurrency: int = 3,
        return_exceptions: bool = True,
    ) -> list[TaskSolution | BaseException]:
        """Обработать пачку задач конкурентно.

        :param items: пары ``(task_id, raw_text)``.
        :param concurrency: сколько запросов держать в полёте одновременно
            (защита от квот Gemini).
        :param return_exceptions: если ``True``, ошибка по одной задаче не роняет
            всю пачку — она возвращается в списке результатов как исключение.
        """
        semaphore = asyncio.Semaphore(max(1, concurrency))

        async def worker(task_id: str | None, raw_text: str) -> TaskSolution:
            async with semaphore:
                return await self.process(raw_text, task_id=task_id)

        tasks = [worker(task_id, raw_text) for task_id, raw_text in items]
        return await asyncio.gather(*tasks, return_exceptions=return_exceptions)
