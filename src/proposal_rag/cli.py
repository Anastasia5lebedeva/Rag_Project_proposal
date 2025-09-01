from __future__ import annotations
import asyncio
from pathlib import Path
import typer

app = typer.Typer(add_completion=False)

@app.command()
def index(path: str):
    p = Path(path)
    # вызови из твоего document_processor разбор и чанкинг
    typer.echo(f"Indexed: {p}")

@app.command()
def query(path: str, q: str, k: int = 10):
    p = Path(path)
    # вызови твой ретрив из vector_search
    res = []
    for r in res:
        typer.echo(r)

if __name__ == "__main__":
    app()