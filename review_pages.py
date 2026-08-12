"""Страницы просмотра базы: разложить задачи перед глазами и отметить лишние.

Отбор упирается в то, что каждую задачу надо увидеть, а ходить по двум тысячам
вкладок в админке — неделя работы. Модуль берёт локальную базу, выкачанную
``umschool.py``, и собирает по каждому прототипу одну HTML-страницу: все задачи
карточками, сгруппированы по темам, с условиями, картинками, ответами и
подсказками.

Страница работает офлайн (кроме картинок, они грузятся с сервера) и ничего
никуда не отправляет. Отметки хранятся в ``localStorage`` браузера, поэтому
случайное обновление вкладки не стирает часовую работу, а кнопка выгрузки
отдаёт CSV, который вставляется в рабочую таблицу.

Задачи не удаляются и не меняются — на выходе только ваши отметки.

Пример::

    python umschool.py --prototype 943 --out output/db.jsonl
    python review_pages.py output/db.jsonl --out output/review
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import re
import webbrowser
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from umschool import UmTask, load_tasks

logger = logging.getLogger(__name__)

#: Куда складывать страницы просмотра.
DEFAULT_OUT_DIR = Path("output/review")

#: Скрипты внутри условия нам не нужны: страница открывается локально, и чужой
#: JS из базы там выполняться не должен.
SCRIPT_PATTERN = re.compile(r"<script\b.*?</script>", re.IGNORECASE | re.DOTALL)

#: Обработчики событий в атрибутах — по той же причине.
HANDLER_PATTERN = re.compile(r"\son\w+\s*=\s*(\"[^\"]*\"|'[^']*'|[^\s>]+)", re.IGNORECASE)


def sanitize(markup: str) -> str:
    """Убрать из разметки исполняемые куски, оставив вёрстку и картинки."""
    cleaned = SCRIPT_PATTERN.sub("", markup or "")
    return HANDLER_PATTERN.sub("", cleaned)


PAGE_CSS = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body { font: 15px/1.55 -apple-system, "Segoe UI", Roboto, sans-serif;
       margin: 0; background: #f4f5f7; color: #16181d; }
header { position: sticky; top: 0; z-index: 10; background: #fff;
         border-bottom: 1px solid #dcdfe4; padding: 12px 20px;
         display: flex; gap: 16px; align-items: center; flex-wrap: wrap; }
header h1 { font-size: 17px; margin: 0; flex: 1 1 auto; }
.counter { font-variant-numeric: tabular-nums; color: #555; }
.counter b { color: #16181d; }
button { font: inherit; padding: 6px 14px; border-radius: 6px; cursor: pointer;
         border: 1px solid #c4c8ce; background: #fff; }
button.primary { background: #2d6cdf; border-color: #2d6cdf; color: #fff; }
main { padding: 20px; max-width: 1100px; margin: 0 auto; }
h2 { font-size: 16px; margin: 28px 0 10px; padding-bottom: 6px;
     border-bottom: 2px solid #dcdfe4; }
h2 .count { color: #777; font-weight: 400; }
.card { background: #fff; border: 1px solid #dcdfe4; border-left: 4px solid #dcdfe4;
        border-radius: 8px; padding: 14px 16px; margin-bottom: 12px; }
.card.dropped { border-left-color: #d9534f; background: #fbf3f3; opacity: .65; }
.card.focused { border-color: #2d6cdf; box-shadow: 0 0 0 2px rgba(45,108,223,.2); }
.card-head { display: flex; gap: 12px; align-items: center; flex-wrap: wrap;
             margin-bottom: 8px; }
.card-head label { display: flex; gap: 6px; align-items: center; cursor: pointer;
                   font-weight: 600; }
.meta { color: #666; font-size: 13px; }
.meta a { color: #2d6cdf; }
.tag { display: inline-block; padding: 1px 8px; border-radius: 10px;
       background: #eef0f3; font-size: 12px; color: #444; }
.condition img { max-width: 100%; height: auto; }
.condition table { border-collapse: collapse; margin: 8px 0; }
.condition td, .condition th { border: 1px solid #999; padding: 3px 10px; }
details { margin-top: 8px; }
summary { cursor: pointer; color: #2d6cdf; font-size: 14px; }
.answer { font-weight: 600; background: #eaf6ea; padding: 2px 10px; border-radius: 4px; }
.note { width: 100%; margin-top: 8px; padding: 6px 8px; border: 1px solid #dcdfe4;
        border-radius: 6px; font: inherit; }
.hint { color: #777; font-size: 13px; margin: 4px 0 0; }
@media (prefers-color-scheme: dark) {
  body { background: #14161a; color: #e6e8eb; }
  header, .card { background: #1c1f25; border-color: #2c3038; }
  .card.dropped { background: #2a1d1e; }
  button { background: #23272e; border-color: #3a3f48; color: #e6e8eb; }
  .tag { background: #2a2e36; color: #c2c6cc; }
  .counter { color: #aab; } .counter b { color: #e6e8eb; }
  .note { background: #23272e; border-color: #3a3f48; color: #e6e8eb; }
}
"""

PAGE_JS = """
const KEY = 'review-' + PROTOTYPE_KEY;
const state = JSON.parse(localStorage.getItem(KEY) || '{}');
const cards = Array.from(document.querySelectorAll('.card'));
let focused = 0;

function apply(card) {
  const id = card.dataset.id;
  const saved = state[id] || {};
  const box = card.querySelector('input[type=checkbox]');
  const note = card.querySelector('.note');
  box.checked = Boolean(saved.dropped);
  if (saved.note) note.value = saved.note;
  card.classList.toggle('dropped', box.checked);
}

function save() {
  localStorage.setItem(KEY, JSON.stringify(state));
  const dropped = Object.values(state).filter((item) => item.dropped).length;
  document.getElementById('dropped').textContent = dropped;
  document.getElementById('kept').textContent = cards.length - dropped;
}

function toggle(card, value) {
  const id = card.dataset.id;
  const box = card.querySelector('input[type=checkbox]');
  box.checked = value === undefined ? !box.checked : value;
  state[id] = Object.assign({}, state[id], { dropped: box.checked });
  card.classList.toggle('dropped', box.checked);
  save();
}

cards.forEach((card, index) => {
  apply(card);
  card.querySelector('input[type=checkbox]').addEventListener('change', (event) => {
    toggle(card, event.target.checked);
  });
  card.querySelector('.note').addEventListener('input', (event) => {
    state[card.dataset.id] = Object.assign({}, state[card.dataset.id],
                                           { note: event.target.value });
    save();
  });
  card.addEventListener('click', () => { focus(index); });
});

function focus(index) {
  if (index < 0 || index >= cards.length) return;
  cards[focused].classList.remove('focused');
  focused = index;
  cards[focused].classList.add('focused');
}

document.addEventListener('keydown', (event) => {
  if (event.target.tagName === 'TEXTAREA' || event.target.tagName === 'INPUT') return;
  if (event.key === 'j' || event.key === 'ArrowDown') {
    focus(focused + 1); cards[focused].scrollIntoView({ block: 'center' });
    event.preventDefault();
  } else if (event.key === 'k' || event.key === 'ArrowUp') {
    focus(focused - 1); cards[focused].scrollIntoView({ block: 'center' });
    event.preventDefault();
  } else if (event.key === 'x' || event.key === ' ') {
    toggle(cards[focused]); event.preventDefault();
  }
});

document.getElementById('hide-dropped').addEventListener('click', (event) => {
  const on = document.body.classList.toggle('hide-dropped');
  event.target.textContent = on ? 'Показать помеченные' : 'Скрыть помеченные';
  document.querySelectorAll('.card.dropped').forEach((card) => {
    card.style.display = on ? 'none' : '';
  });
});

document.getElementById('export').addEventListener('click', () => {
  const rows = [['Айди задания', 'Ссылка на задание', 'Номер задания', 'Тема',
                 'Неактуально', 'Заметка']];
  cards.forEach((card) => {
    const saved = state[card.dataset.id] || {};
    rows.push([card.dataset.id, card.dataset.url, card.dataset.prototype,
               card.dataset.topic, saved.dropped ? 'TRUE' : '',
               (saved.note || '').replace(/\\s+/g, ' ')]);
  });
  const csv = rows.map((row) =>
    row.map((cell) => '"' + String(cell).replace(/"/g, '""') + '"').join(',')
  ).join('\\n');
  const blob = new Blob(['\\ufeff' + csv], { type: 'text/csv;charset=utf-8' });
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = 'review-' + PROTOTYPE_KEY + '.csv';
  link.click();
});

save();
focus(0);
"""


def render_card(task: UmTask) -> str:
    """Собрать карточку одной задачи."""
    flags: list[str] = []
    if task.difficulty:
        flags.append(f'<span class="tag">{html.escape(task.difficulty)}</span>')
    if task.checked_by_expert:
        flags.append('<span class="tag">проверено экспертом</span>')
    if task.for_generation:
        flags.append('<span class="tag">для генерации</span>')
    if not task.images:
        flags.append('<span class="tag">без картинки</span>')

    answers = ", ".join(task.answers) or "—"
    tips = "".join(f"<li>{html.escape(tip)}</li>" for tip in task.tips)
    tips_block = (
        f"<details><summary>Подсказки ({len(task.tips)})</summary><ol>{tips}</ol></details>"
        if task.tips
        else ""
    )
    solution_block = (
        f"<details><summary>Решение</summary>"
        f'<div class="condition">{sanitize(task.solution_html)}</div></details>'
        if task.solution_html
        else '<p class="hint">Решения нет</p>'
    )

    return f"""
<div class="card" data-id="{task.id}" data-url="{html.escape(task.url)}"
     data-prototype="{html.escape(task.prototype_title)}"
     data-topic="{html.escape(task.topic_title)}">
  <div class="card-head">
    <label><input type="checkbox"> неактуально</label>
    <span class="meta">id {task.id} · <a href="{html.escape(task.url)}"
      target="_blank" rel="noopener">открыть в админке</a></span>
    <span class="meta">ответ: <span class="answer">{html.escape(answers)}</span></span>
    {" ".join(flags)}
  </div>
  <div class="condition">{sanitize(task.condition_html)}</div>
  {solution_block}
  {tips_block}
  <textarea class="note" rows="1" placeholder="заметка — почему сняли или оставили"></textarea>
</div>"""


def render_page(prototype: str, tasks: Sequence[UmTask], key: str) -> str:
    """Собрать страницу просмотра одного прототипа."""
    by_topic: dict[str, list[UmTask]] = defaultdict(list)
    for task in tasks:
        by_topic[task.topic_title or "тема не проставлена"].append(task)

    sections: list[str] = []
    for topic in sorted(by_topic):
        group = by_topic[topic]
        cards = "".join(render_card(task) for task in group)
        sections.append(
            f'<h2>{html.escape(topic)} <span class="count">— {len(group)}</span></h2>{cards}'
        )

    return f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(prototype)} — просмотр</title>
<style>{PAGE_CSS}</style></head><body>
<header>
  <h1>{html.escape(prototype)}</h1>
  <span class="counter">оставлено <b id="kept">0</b> · снято <b id="dropped">0</b>
    из {len(tasks)}</span>
  <button id="hide-dropped">Скрыть помеченные</button>
  <button id="export" class="primary">Выгрузить CSV</button>
</header>
<main>
  <p class="hint">Отметки сохраняются в браузере сами. Клавиши: <b>j</b>/<b>k</b> —
    следующая и предыдущая задача, <b>x</b> или пробел — снять или вернуть.</p>
  {"".join(sections)}
</main>
<script>const PROTOTYPE_KEY = {json.dumps(key)};{PAGE_JS}</script>
</body></html>"""


def build_pages(
    tasks: Sequence[UmTask], out_dir: Path, *, suffix: str = ""
) -> list[Path]:
    """Собрать по странице на каждый прототип.

    :param suffix: добавка к имени файла и ключу хранения отметок. Нужна, чтобы
        просмотр своих задач и просмотр кандидатов на замену не перетирали
        отметки друг друга.
    :returns: пути к созданным страницам.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_prototype: dict[tuple[int, str], list[UmTask]] = defaultdict(list)
    for task in tasks:
        key = (task.prototype_order or 99, task.prototype_title or "без прототипа")
        by_prototype[key].append(task)

    created: list[Path] = []
    for (order, title), group in sorted(by_prototype.items()):
        key = f"{order}-{group[0].prototype_code or 'proto'}{suffix}"
        label = f"{order}. {title}"
        if suffix == "-others":
            label += " — кандидаты на замену"
        page = render_page(label, group, key)
        path = out_dir / f"prototype-{key}.html"
        path.write_text(page, encoding="utf-8")
        created.append(path)
        logger.info("%s — %s задач → %s", title, len(group), path)
    return created


def read_task_ids(path: Path) -> set[int]:
    """Достать идентификаторы задач из выгрузки рабочей таблицы.

    Понимает CSV и TSV: если есть колонка «Айди задания» (или ``id``), берутся
    значения из неё, иначе — все целые числа подходящей длины из строки. Второй
    путь нужен, потому что таблицу выгружают по-разному, а ошибиться колонкой
    легко.

    :raises ValueError: файл не читается или идентификаторов в нём нет.
    """
    try:
        lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        raise ValueError(f"Не удалось прочитать {path}: {exc}") from exc

    if not lines:
        raise ValueError(f"{path}: файл пуст")

    separator = "\t" if lines[0].count("\t") >= lines[0].count(",") else ","
    header = [cell.strip().strip('"').lower() for cell in lines[0].split(separator)]
    column = next(
        (
            index
            for index, name in enumerate(header)
            if "айди" in name or name in {"id", "task_id"}
        ),
        None,
    )

    ids: set[int] = set()
    for line in lines[1:] if column is not None else lines:
        if column is not None:
            cells = line.split(separator)
            if column < len(cells):
                value = cells[column].strip().strip('"')
                if value.isdigit():
                    ids.add(int(value))
            continue
        # Колонки не нашли — подбираем любые числа, похожие на идентификаторы.
        ids.update(int(found) for found in re.findall(r"\b\d{5,9}\b", line))

    if not ids:
        raise ValueError(f"{path}: не нашёл идентификаторов задач")
    return ids


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Разобрать аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description="Страницы просмотра базы: отметить неактуальные задачи."
    )
    parser.add_argument("database", type=Path, help="JSONL из umschool.py")
    parser.add_argument(
        "--table",
        type=Path,
        help="Выгрузка рабочей таблицы (CSV или TSV) — из неё берутся id задач",
    )
    parser.add_argument(
        "--mode",
        choices=("mine", "others", "all"),
        default=None,
        help="Что показывать: mine — только задачи из таблицы (по умолчанию, "
        "если она задана), others — только остальную базу как кандидатов на "
        "замену, all — всё подряд",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Куда сложить страницы (по умолчанию {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--open", action="store_true", help="Открыть первую страницу в браузере"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Точка входа CLI. Возвращает код возврата процесса."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args(argv)

    if not args.database.exists():
        logger.error("Нет файла базы %s — сначала выкачайте: python umschool.py", args.database)
        return 1

    tasks = load_tasks(args.database)
    if not tasks:
        logger.error("База пуста")
        return 1

    mode = args.mode or ("mine" if args.table else "all")
    if mode != "all":
        if not args.table:
            logger.error("Режим «%s» требует --table с выгрузкой таблицы", mode)
            return 1
        try:
            table_ids = read_task_ids(args.table)
        except ValueError as exc:
            logger.error("%s", exc)
            return 1

        total = len(tasks)
        if mode == "mine":
            tasks = [task for task in tasks if task.id in table_ids]
            logger.info(
                "Из таблицы: %s задач(и) из %s в базе; в таблице всего %s",
                len(tasks), total, len(table_ids),
            )
            missing = len(table_ids) - len(tasks)
            if missing > 0:
                logger.warning(
                    "%s задач(и) из таблицы нет в выкачанной базе — возможно, "
                    "не хватает прототипов",
                    missing,
                )
        else:
            tasks = [task for task in tasks if task.id not in table_ids]
            logger.info(
                "Кандидаты на замену: %s задач(и) из %s (не входят в таблицу)",
                len(tasks), total,
            )

    if not tasks:
        logger.error("После фильтра не осталось задач")
        return 1

    pages = build_pages(tasks, args.out, suffix="" if mode == "all" else f"-{mode}")
    print(f"\nГотово: {len(pages)} страниц(ы) в {args.out}")
    for path in pages:
        print(f"  {path}")

    if args.open and pages:
        webbrowser.open(pages[0].resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
