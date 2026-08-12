"""Интеграция с Gemini API: превращает сырой текст задачи в структурированный JSON.

Модуль отвечает ровно за один шаг пайплайна: получить на вход текст условия
(как его вернул ``scraper.py``) и вернуть провалидированный объект
:class:`TaskSolution` с полями ``task_id``, ``condition``, ``solution_text``,
``answer``, ``hints``.

Ключевые решения:

* Используется SDK ``google-genai`` (пакет ``google-genai``, импорт
  ``from google import genai``) — преемник закрытого ``google-generativeai``.
* Ответ ограничен схемой (``response_schema`` + ``response_mime_type``): API сам
  следит за набором полей и типами. Мы всё равно не доверяем ответу и парсим его
  защитно (см. :func:`parse_llm_json`) — схема снимает частые сбои, но не
  отменяет проверку.
* Все сетевые вызовы асинхронные (``client.aio``), поэтому пачку задач можно
  обрабатывать конкурентно через :meth:`GeminiProcessor.process_many`.
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

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Константы
# --------------------------------------------------------------------------- #

#: Модель по умолчанию; переопределяется переменной окружения ``GEMINI_MODEL``.
#: Для разборов ЕГЭ важнее рассуждения, чем скорость, поэтому pro.
DEFAULT_MODEL = "gemini-2.5-pro"

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

#: Инструкция для переписывания условия «под себя».
#:
#: Нужна, чтобы разбор на сайте школы не был дословной копией чужого условия.
#: Переписывание обязано сохранять тип задания и способ решения — иначе выйдет
#: другая задача, а не переформулированная.
REWRITE_INSTRUCTION = """\
Перепиши условие задачи своими словами так, чтобы текст заметно отличался от
исходного, но задача осталась той же по типу, сложности и способу решения.

ЧТО МЕНЯТЬ МОЖНО И НУЖНО:
- формулировки предложений, порядок их следования;
- имена персонажей;
- порядок перечисления переменных (например, w, x, y, z → x, y, z, w);
- порядок столбцов или строк в таблице, если это не меняет сути;
- нейтральные детали оформления.

ЧТО МЕНЯТЬ НЕЛЬЗЯ:
- тип задания и проверяемое умение;
- логическую структуру и сложность;
- числовые данные, если от них зависит ответ.

КРИТИЧЕСКИ ВАЖНО:
Если перестановка переменных или столбцов меняет правильный ответ — ПЕРЕСЧИТАЙ
ответ под новую формулировку. Ответ должен соответствовать именно тому условию,
которое ты написал. Решение и подсказки тоже перепиши под новую формулировку,
чтобы они ссылались на актуальный порядок переменных.

Верни полный JSON-объект того же формата со всеми пятью полями.
"""

#: Заголовок блока с образцом оформления.
STYLE_HINT = """\
Ниже приведён ОБРАЗЕЦ того, как должен выглядеть разбор: его структура, стиль
изложения и уровень подробности. Следуй образцу по форме подачи, но содержание
бери из своей задачи — не копируй числа и выводы образца.

--- ОБРАЗЕЦ ---
{sample}
--- КОНЕЦ ОБРАЗЦА ---
"""

#: Схема ответа для API. Дублирует требования системного промпта на уровне,
#: который модель нарушить не может: набор полей, их типы и обязательность.
RESPONSE_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    required=["task_id", "condition", "solution_text", "answer", "hints"],
    properties={
        "task_id": types.Schema(
            type=types.Type.STRING, description="Идентификатор задачи"
        ),
        "condition": types.Schema(
            type=types.Type.STRING, description="Очищенное условие задачи"
        ),
        "solution_text": types.Schema(
            type=types.Type.STRING, description="Пошаговое решение"
        ),
        "answer": types.Schema(
            type=types.Type.STRING, description="Финальный ответ без пояснений"
        ),
        "hints": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(type=types.Type.STRING),
            description="Подсказки, не раскрывающие ответ",
        ),
    },
)


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

        self._client = genai.Client(api_key=key)
        self._config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
            # Просим API отдавать именно JSON заданной формы — это снимает
            # markdown-ограждения, болтовню вокруг ответа и пропуск полей.
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
        )
        logger.debug("GeminiProcessor инициализирован (модель=%s)", self.model_name)

    # -- внутреннее ---------------------------------------------------------- #

    @staticmethod
    def _build_prompt(
        raw_text: str, task_id: str | None, style_reference: str = ""
    ) -> str:
        """Собрать пользовательскую часть промпта.

        :param style_reference: образец оформления разбора. Если задан, модель
            получает его вместе с задачей и подражает форме подачи — так первый
            же вариант выходит в нужном стиле, без правок вручную.
        """
        header = f"ID задачи: {task_id}\n\n" if task_id else ""
        prompt = f"{header}Сырой текст задачи:\n<<<\n{raw_text.strip()}\n>>>"
        if style_reference.strip():
            prompt = f"{STYLE_HINT.format(sample=style_reference.strip())}\n\n{prompt}"
        return prompt

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
                response = await self._client.aio.models.generate_content(
                    model=self.model_name, contents=prompt, config=self._config
                )
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

    async def process(
        self,
        raw_text: str,
        *,
        task_id: str | None = None,
        style_reference: str = "",
    ) -> TaskSolution:
        """Обработать одну задачу.

        :param raw_text: сырой текст условия из ``scraper.py``.
        :param task_id: известный идентификатор задачи; подставится в результат,
            если модель не вернёт свой.
        :param style_reference: образец оформления разбора (см.
            :meth:`_build_prompt`).
        :raises LLMRequestError: сбой обращения к API.
        :raises LLMResponseError: ответ не удалось распарсить/провалидировать.
        """
        if not raw_text or not raw_text.strip():
            raise LLMResponseError("На вход подан пустой текст задачи")

        logger.info("Отправляю задачу %s в Gemini", task_id or "<без id>")
        raw_response = await self._generate(
            self._build_prompt(raw_text, task_id, style_reference)
        )
        solution = parse_llm_json(raw_response, fallback_task_id=task_id)
        logger.info("Задача %s обработана", solution.task_id or "<без id>")
        return solution

    def start_session(
        self, raw_text: str, solution: TaskSolution, *, style_reference: str = ""
    ) -> "SolutionSession":
        """Открыть диалог с моделью вокруг уже полученного разбора.

        Диалог нужен, чтобы править решение репликами («перепиши через
        перебор», «добавь второй способ»), а не переспрашивать задачу с нуля:
        модель помнит и условие, и предыдущие версии разбора.
        """
        return SolutionSession(
            self,
            self._build_prompt(raw_text, solution.task_id, style_reference),
            solution,
        )

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


# --------------------------------------------------------------------------- #
# Диалог по одной задаче
# --------------------------------------------------------------------------- #


class SolutionSession:
    """Многоходовой диалог с моделью вокруг одного разбора.

    Создаётся через :meth:`GeminiProcessor.start_session`. Модель видит исходную
    задачу и все предыдущие версии разбора, поэтому правки формулируются
    репликами: «сделай через перебор в Python», «добавь второй способ»,
    «вот образец, приведи к такому виду».

    Каждый ответ проходит ту же валидацию, что и первичный разбор, поэтому
    :attr:`solution` всегда остаётся цельным :class:`TaskSolution`.
    """

    def __init__(
        self, processor: GeminiProcessor, prompt: str, solution: TaskSolution
    ) -> None:
        self._processor = processor
        self._prompt = prompt
        self.solution = solution
        self.history: list[str] = []
        self._chat: Any = None

    def _ensure_chat(self) -> Any:
        """Создать чат, подставив в историю исходную задачу и текущий разбор."""
        if self._chat is None:
            self._chat = self._processor._client.aio.chats.create(
                model=self._processor.model_name,
                config=self._processor._config,
                history=[
                    types.Content(
                        role="user", parts=[types.Part.from_text(text=self._prompt)]
                    ),
                    types.Content(
                        role="model",
                        parts=[types.Part.from_text(text=self.solution.to_json())],
                    ),
                ],
            )
        return self._chat

    async def send(self, message: str) -> TaskSolution:
        """Отправить произвольную реплику и получить обновлённый разбор.

        :raises LLMRequestError: сбой обращения к API.
        :raises LLMResponseError: ответ не удалось распарсить/провалидировать.
        """
        if not message.strip():
            raise LLMResponseError("Пустая реплика — нечего отправлять модели")

        chat = self._ensure_chat()
        try:
            response = await chat.send_message(message)
        except Exception as exc:  # SDK бросает разнородные исключения
            raise LLMRequestError(f"Правка не удалась: {exc}") from exc

        text = GeminiProcessor._extract_text(response)
        self.solution = parse_llm_json(
            text, fallback_task_id=self.solution.task_id
        )
        self.history.append(message)
        return self.solution

    async def refine(self, instruction: str, sample: str = "") -> TaskSolution:
        """Поправить разбор по указанию, при желании показав образец.

        :param instruction: что именно изменить.
        :param sample: образец оформления — «вот как должно быть».
        """
        message = instruction.strip()
        if sample.strip():
            message = f"{message}\n\n{STYLE_HINT.format(sample=sample.strip())}"
        return await self.send(message)

    async def rewrite(self, extra_instruction: str = "") -> TaskSolution:
        """Переписать условие своими словами и пересчитать под него ответ.

        Нужно, чтобы разбор не был дословной копией чужого условия. Результат
        обязательно проверяйте независимым решением: см.
        :func:`verify_rewrite`.
        """
        message = REWRITE_INSTRUCTION
        if extra_instruction.strip():
            message = f"{message}\n\nДополнительно: {extra_instruction.strip()}"
        return await self.send(message)


async def verify_rewrite(
    processor: GeminiProcessor, solution: TaskSolution
) -> tuple[bool, str]:
    """Проверить переписанное условие независимым решением.

    После переписывания ответ с сайта эталоном быть перестаёт: у новой
    формулировки он может быть другим. Поэтому переписанное условие решается
    заново — отдельным запросом, без контекста диалога, чтобы модель не
    подсматривала собственный предыдущий ответ.

    :returns: пара ``(сошлось_ли, ответ_проверки)``.
    """
    control = await processor.process(solution.condition, task_id=solution.task_id)
    matched = control.answer.strip().casefold() == solution.answer.strip().casefold()
    if not matched:
        logger.warning(
            "Проверка переписанного условия задачи %s не сошлась: «%s» против «%s»",
            solution.task_id,
            solution.answer,
            control.answer,
        )
    return matched, control.answer
