from __future__ import annotations

from pathlib import Path
import os
import json
from typing import List, Dict, Any, Optional
import asyncio
import random

import httpx

# === PROJECT ROOT ===
PROJECT_DIR = Path(os.getenv("PROJECT_DIR", "/home/anastasia/PycharmProjects/Rag_Project_proposal"))

# Structure expectations (src-layout)
SRC_DIR = PROJECT_DIR / "src"
DB_DIR = PROJECT_DIR / "database"
PROMPTS_DIR = PROJECT_DIR / "prompts"
CONFIG_DIR = SRC_DIR / "config"
SERVICES_DIR = SRC_DIR / "services"
MODELS_DIR = SRC_DIR / "models"
API_DIR = SRC_DIR / "api"
TESTS_DIR = SRC_DIR / "tests"
PYPROJECT = PROJECT_DIR / "pyproject.toml"
POETRY_LOCK = PROJECT_DIR / "poetry.lock"
DOCKER_COMPOSE = PROJECT_DIR / "docker-compose.yml"
MYPY_INI = PROJECT_DIR / "mypy.ini"
PYTEST_INI = PROJECT_DIR / "pytest.ini"
ENV_FILES = [PROJECT_DIR / ".env", PROJECT_DIR / ".env.stage", PROJECT_DIR / ".env.local"]

# Output
OUT_DIR = PROJECT_DIR / "reviews"
OUT_FILE = OUT_DIR / "project_refactor_audit.md"


BASE_URL = "http://api.comrade.168.119.237.99.sslip.io/v1"
API_KEY = "06662fe1-bed2-496e-adbc-c2eea331f151"
MODEL = "gpt-4.1"
DATABASE_URL = "postgresql://cortex:prodpassword@prod-db:5432/cortex_rag"
EMB_MODEL = "intfloat/multilingual-e5-large"
DEFAULT_QUERY_PATH = "/app/data/query.docx"
TOPK = 20

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

# Limits
TREE_LIMIT = 900  # max lines for tree preview


def _extract_text_from_response(data: Dict[str, Any]) -> str:
    """
    Extract text from OpenAI-compatible or adapter-like responses.
    Return non-empty string or raise.
    """
    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid provider payload type: {type(data)}")

    if data.get("error"):
        raise RuntimeError(f"Provider error: {data['error']}")

    choices = data.get("choices", [])
    if isinstance(choices, list) and choices:
        msg = choices[0].get("message")
        if isinstance(msg, dict):
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                return content
        # some providers put text/content directly on choice
        for key in ("content", "text"):
            val = choices[0].get(key)
            if isinstance(val, str) and val.strip():
                return val

    # fallbacks used by some proxies
    for key in ("output_text", "response", "content"):
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            return val

    raise RuntimeError(f"Cannot extract content; payload preview: {json.dumps(data, ensure_ascii=False)[:2000]}")


def _read_or(name: str, path: Path, limit: Optional[int] = None) -> str:
    if not path.exists():
        return f"# {name} — {path}\n<NOT FOUND>\n"
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if limit and len(text) > limit:
            text = text[:limit] + "\n... (truncated)\n"
        return f"# {name} — {path}\n```text\n{text}\n```\n"
    except Exception as e:
        return f"# {name} — {path}\n<read error: {e}>\n"


def _folder_tree(root: Path, limit_lines: int = 1200) -> str:
    lines: List[str] = []

    def walk(d: Path, prefix: str = "") -> None:
        try:
            entries = sorted(d.iterdir(), key=lambda x: (x.is_file(), x.name))
        except Exception:
            return
        for i, e in enumerate(entries):
            if e.name.startswith(".") or e.name in {"__pycache__", ".venv", ".git", ".mypy_cache"} or e.suffix == ".egg-info":
                continue
            connector = "└── " if i == len(entries) - 1 else "├── "
            lines.append(f"{prefix}{connector}{e.name}")
            if len(lines) >= limit_lines:
                lines.append("... (truncated)")
                return
            if e.is_dir():
                walk(e, prefix + ("    " if i == len(entries) - 1 else "│   "))

    lines.append(root.name)
    walk(root)
    return "# TREE\n```text\n" + "\n".join(lines) + "\n```\n"


def _collect_glob(root: Path, patterns: tuple[str, ...], title: str, hard_limit: Optional[int] = None) -> str:
    if not root.exists():
        return f"# {title}\n<root not found: {root}>\n"
    blocks: List[str] = []
    for pat in patterns:
        for p in sorted(root.rglob(pat)):
            if any(part in p.parts for part in (".git", ".venv", "__pycache__")):
                continue
            try:
                txt = p.read_text(encoding="utf-8", errors="ignore")
                if hard_limit and len(txt) > hard_limit:
                    txt = txt[:hard_limit] + "\n... (truncated)\n"
            except Exception as e:
                txt = f"<<read error: {e}>>"
            rel = p.relative_to(PROJECT_DIR)
            fence = "python" if p.suffix == ".py" else ("sql" if p.suffix == ".sql" else "text")
            blocks.append(f"### {rel}\n```{fence}\n{txt}\n```\n")
    return f"# {title}\n" + ("\n".join(blocks) if blocks else "<none>\n")


def build_system_message() -> str:
    return (
        "Ты — строгий техлид/аудитор проекта Rag_Project_proposal. "
        "Работаешь как архитектор: минимальные правки, максимум эффекта, без поломок.\n"
        "\n"
        "ТВОЯ МИССИЯ:\n"
        "• Проанализировать ВЕСЬ код и конфиги, собранные в сообщении пользователя, и выдать детальный аудит.\n"
        "• Предложить конкретные правки (патчи), которые можно сразу применить.\n"
        "• Не выдумывать артефакты, которых нет в присланном контенте. Любые догадки явно помечай как HYPOTHESIS.\n"
        "\n"
        "ОГРАНИЧЕНИЯ И СТАНДАРТЫ:\n"
        "• Проект должен иметь src-layout и следующую логическую структуру:\n"
        "  src/{config,models,services,tests,api} и без монолитных god-functions в main.\n"
        "• Конфигурация только через .env (+ pydantic BaseSettings). Никаких «магических чисел» и хардкода.\n"
        "• В исходниках (.py) не допускается кириллица и не-ASCII символы (строки промптов тоже выносятся наружу).\n"
        "• Логирование: структурированное (logging), уровни, хендлеры; в API — корректные коды ошибок.\n"
        "• Валидация входных данных (pydantic/FastAPI). Явная обработка исключений.\n"
        "• Перформанс: кэш эмбеддингов + индексы БД на горячих полях. Отдельно отметить места N+1, лишние I/O, повторные вызовы LLM.\n"
        "• Промпты вынести в файлы (prompts/ *.yaml|*.md|*.txt), обеспечить загрузку/версирование.\n"
        "\n"
        "ЧТО ПРОВЕРЯЕШЬ И ЧТО ДОЛЖЕН СДЕЛАТЬ:\n"
        "1) РЕФАКТОРИНГ АРХИТЕКТУРЫ\n"
        "   - Фактическое дерево /src: где отсутствуют каталоги, где путаница в слоях.\n"
        "   - services/: наличие чётких модулей document_processor.py, vector_search.py, llm_client.py.\n"
        "   - api/: отделение web-слоя (FastAPI/routers) от бизнес-логики.\n"
        "   - models/: DTO/схемы/доменные модели отдельно от БД-слоя.\n"
        "   - Исключить дублирование: выделить общие утилиты.\n"
        "\n"
        "2) ОПТИМИЗАЦИЯ ПРОИЗВОДИТЕЛЬНОСТИ\n"
        "   - Кэш эмбеддингов: предложить реализацию (ключ = model+hash(content), хранение: JSONB/таблица/файл-KV), TTL/инвалидация.\n"
        "   - Индексы БД: перечислить горячие поля, предложить точные CREATE INDEX и оправдать их.\n"
        "   - Для LLM: отловить повторные одинаковые запросы, предусмотреть memoization.\n"
        "\n"
        "3) ИНФРАСТРУКТУРА\n"
        "   - Логирование: единый инициализатор, формат, уровни, ротация/stdout для контейнеров.\n"
        "   - Обработка ошибок: единый middleware для API; try/except в сервисах с явной типизацией ошибок.\n"
        "   - Валидация входа: pydantic-модели, строгие типы, явные ошибки 4xx.\n"
        "   - Конфигурация: BaseSettings, AliasChoices для .env, отсутствие хардкода; показать расхождения env↔код.\n"
        "   - Промпты: убрать из кода, описать предложенную структуру prompts/ и схему загрузки.\n"
        "\n"
        "4) ДЕКОМПОЗИЦИЯ MAIN\n"
        "   - Найти монолитные entry-points и расписать, на какие функции их разрезать (пошаговый пайплайн).\n"
        "\n"
        "СТРОГИЙ ФОРМАТ ОТВЕТА:\n"
        "## ARCHITECTURE_AUDIT\n"
        "— Конкретные проблемы по слоям, ссылки на файлы/функции, краткие примеры до/после.\n"
        "\n"
        "## PERFORMANCE\n"
        "— План кэша эмбеддингов (ключ, структура хранения, API), список индексов с точными DDL и объяснением.\n"
        "\n"
        "## INFRASTRUCTURE\n"
        "— Логирование, обработка ошибок, валидация, конфиги (.env↔Settings), вынос промптов.\n"
        "\n"
        "## MAIN_DECOMPOSITION\n"
        "— Какие main/скрипты распилить, итоговый пайплайн шагами.\n"
        "\n"
        "## PATCHES\n"
        "— Набор применимых патчей (unified diff) ВНИЗУ, с реальными путями файлов и минимальными правками.\n"
        "— Если файла нет — сначала # FILE_TO_CREATE: <path> c кратким содержимым.\n"
        "\n"
        "## CHECKLIST\n"
        "— Список шагов для внедрения (1–2 дня, 1 неделя), метрики и быстрая проверка успеха.\n"
        "\n"
        "ВАЖНО:\n"
        "— Опирайся только на присланный контент. Без галлюцинаций. Любые догадки = HYPOTHESIS.Отвечай на русском\n"
    )


def build_user_message() -> str:
    parts: List[str] = []

    # Tree + key configs
    parts.append(_folder_tree(PROJECT_DIR))
    parts.append(_read_or("pyproject.toml", PYPROJECT))
    parts.append(_read_or("poetry.lock", POETRY_LOCK, limit=120000))
    parts.append(_read_or("docker-compose.yml", DOCKER_COMPOSE))
    parts.append(_read_or("mypy.ini", MYPY_INI))
    parts.append(_read_or("pytest.ini", PYTEST_INI))
    for env_path in ENV_FILES:
        parts.append(_read_or(env_path.name, env_path))

    # Code and SQL
    parts.append(_collect_glob(SRC_DIR, ("*.py",), "SRC_PY"))
    parts.append(_collect_glob(DB_DIR, ("*.sql",), "DATABASE_SQL"))
    parts.append(_collect_glob(PROMPTS_DIR, ("*.yaml", "*.yml", "*.md", "*.txt"), "PROMPTS_FILES"))

    # Explicit subfolders
    parts.append(_collect_glob(CONFIG_DIR, ("*.py",), "CONFIG_DIR"))
    parts.append(_collect_glob(SERVICES_DIR, ("*.py",), "SERVICES_DIR"))
    parts.append(_collect_glob(MODELS_DIR, ("*.py",), "MODELS_DIR"))
    parts.append(_collect_glob(API_DIR, ("*.py",), "API_DIR"))
    parts.append(_collect_glob(TESTS_DIR, ("*.py",), "TESTS_DIR"))

    # Non-ASCII hint
    parts.append(
        "# NON_ASCII_CHECK\n"
        "Please detect any non-ASCII symbols in .py sources and propose patches to externalize them into prompts/.\n"
    )

    return "\n\n".join(parts)


async def call_gpt(
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int = 5500,
    timeout: float = 90.0,
    max_retries: int = 3,
    retry_backoff_base: float = 0.8,
) -> str:
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")

    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": float(min(max(temperature, 0.0), 2.0)),
        "max_tokens": int(min(max(max_tokens, 16), 6000)),
    }

    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            async with httpx.AsyncClient(
                base_url=BASE_URL,
                headers=HEADERS,
                timeout=httpx.Timeout(timeout),
            ) as client:
                resp = await client.post("/chat/completions", json=payload)
                if resp.is_error:
                    raise RuntimeError(f"Provider HTTP {resp.status_code}. Body: {resp.text[:9000]}")

                try:
                    data = resp.json()
                except json.JSONDecodeError as e:
                    raise RuntimeError(f"Provider returned non-JSON. Preview: {resp.text[:9000]}") from e

                text = _extract_text_from_response(data)
                if not text:
                    raise RuntimeError(f"Empty content extracted. Raw preview: {json.dumps(data)[:9000]}")

                return text

        except (httpx.TimeoutException, httpx.RemoteProtocolError, RuntimeError) as e:
            last_error = e
            if attempt >= max_retries:
                break
            sleep_s = retry_backoff_base * (2 ** (attempt - 1)) * (1 + 0.25 * random.random())
            await asyncio.sleep(sleep_s)

    assert last_error is not None
    raise last_error


async def main() -> None:
    system = build_system_message()
    user = build_user_message()

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    text = await call_gpt(
        messages=messages,
        temperature=0.0,
        max_tokens=5500,
        timeout=180.0,
        max_retries=3,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(text, encoding="utf-8")
    print(f"Report saved: {OUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())

