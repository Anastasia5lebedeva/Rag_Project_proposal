from __future__ import annotations

import asyncio
import inspect
import json
from pathlib import Path
from typing import Any, Iterable, List, Union

import typer

# Реальные сервисы
from proposal_rag.services.document_processor import smart_chunk
from proposal_rag.services.vector_search import search_hybrid  # sync или async — не важно

app = typer.Typer(add_completion=False, no_args_is_help=True)


# ---- утилиты ---------------------------------------------------------------

def _read_text(path: Union[str, Path]) -> str:
    p = Path(path).expanduser().resolve()
    if not p.exists() or not p.is_file():
        raise typer.BadParameter(f"Файл не найден: {p}")
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        raise typer.BadParameter(f"Не удалось прочитать файл {p}: {e}") from e


def _dump_jsonl(items: Iterable[Any], out_path: Union[str, Path]) -> Path:
    out = Path(out_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for it in items:
            line = json.dumps(it, ensure_ascii=False)
            f.write(line + "\n")
    return out


async def _maybe_await(obj):
    """Поддержка как sync, так и async search_hybrid."""
    if inspect.iscoroutine(obj):
        return await obj
    return obj


# ---- команды ---------------------------------------------------------------

@app.command(help="Разбить текст на чанки и (опц.) сохранить в JSONL.")
def index(
    path: str = typer.Argument(..., help="Путь к входному .txt/.md/.docx (уже извлечённый текст)."),
    out: Path = typer.Option(None, "--out", "-o", help="Путь для сохранения чанков в JSONL."),
    max_tokens: int = typer.Option(1600, help="Максимум токенов на чанк (≈ симв/4)."),
    overlap_tokens: int = typer.Option(250, help="Оверлап между чанками, в токенах."),
    min_tokens: int = typer.Option(150, help="Минимум токенов для сохранения чанка."),
):
    text = _read_text(path)
    chunks: List[str] = smart_chunk(
        text=text,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        min_tokens=min_tokens,
    )
    typer.echo(f"Indexed: {Path(path).resolve()} → {len(chunks)} chunks")

    if out:
        out_path = _dump_jsonl(({"chunk_index": i, "text": c} for i, c in enumerate(chunks)), out)
        typer.echo(f"Saved JSONL: {out_path}")


@app.command(help="Сделать гибридный поиск по индексу.")
def query(
    q: str = typer.Option(..., "--q", "-q", help="Текст запроса."),
    k: int = typer.Option(10, "--k", "-k", help="Количество результатов."),
    pretty: bool = typer.Option(True, "--pretty/--no-pretty", help="Форматированный вывод."),
):
    try:
        # поддержка и sync, и async реализации search_hybrid
        result = asyncio.run(_maybe_await(search_hybrid(q, k)))  # type: ignore[arg-type]
    except RuntimeError as e:
        # если уже есть активный цикл (напр. из uvloop), fallback
        if "asyncio.run() cannot be called from a running event loop" in str(e):
            async def _runner():
                return await _maybe_await(search_hybrid(q, k))
            result = asyncio.get_event_loop().run_until_complete(_runner())
        else:
            raise
    except Exception as e:
        typer.echo(f"Error: {e}")
        raise typer.Exit(code=1)

    # Универсальный дружественный вывод
    def _echo_row(row: Any):
        if isinstance(row, (list, tuple)):
            typer.echo(" | ".join(str(x) for x in row))
        elif isinstance(row, dict):
            typer.echo(json.dumps(row, ensure_ascii=False, indent=2 if pretty else None))
        else:
            typer.echo(str(row))

    if not result:
        typer.echo("No results.")
        return

    # Некоторые реализации возвращают (rows, sql_debug)
    if isinstance(result, (list, tuple)) and len(result) == 2 and isinstance(result[1], str):
        rows, debug = result  # type: ignore[assignment]
        typer.echo(f"-- debug: {debug}")
        for r in rows:
            _echo_row(r)
        return

    # Обычный список результатов
    if isinstance(result, list):
        for r in result:
            _echo_row(r)
        return

    # Единичный результат
    _echo_row(result)


# ---- точка входа -----------------------------------------------------------

if __name__ == "__main__":
    app()

















