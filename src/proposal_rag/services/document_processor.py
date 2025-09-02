from __future__ import annotations
from pathlib import Path
import os
import re
import unicodedata
import hashlib
from typing import Dict, List, Optional

PROMPTS_FILE = Path(os.environ["PROMPTS_FILE"]).resolve()

CHUNK_PROFILE = os.environ["CHUNK_PROFILE"]
CHUNK_MAX_TOKENS = int(os.environ["CHUNK_MAX_TOKENS"])
CHUNK_OVERLAP_TOKENS = int(os.environ["CHUNK_OVERLAP_TOKENS"])
CHUNK_MIN_TOKENS = int(os.environ["CHUNK_MIN_TOKENS"])
CHUNK_MAX_CHARS = int(os.environ["CHUNK_MAX_CHARS"]) or None
PATTERN_HDR = os.environ["PATTERN_HDR"]
PATTERN_BULLET = os.environ["PATTERN_BULLET"]



NBSP = "\u00A0"
ZWSP = "\u200B"
BOM = "\ufeff"
WS_RE = re.compile(r"[ \t\u00A0]+")
MULTINL_RE = re.compile(r"\n{3,}")
CTRL_RE = re.compile(r"[\u0000-\u0008\u000B\u000C\u000E-\u001F]+")
HYPHEN_BREAK_RE = re.compile(r"(?<=\w)-\s*\n\s*(?=\w)")
SOFT_HYPHEN_RE = re.compile(r"\u00AD")

UNICODE_FIXES = {
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-",
    "\u2018": "'", "\u2019": "'", "\u201C": '"', "\u201D": '"',
}


def _read_patterns_kv(path: Path) -> Dict[str, str]:
    if not path.is_file():
        return {}
    patterns: Dict[str, str] = {}
    cur_key: Optional[str] = None
    cur_lines: List[str] = []
    def _flush() -> None:
        nonlocal cur_key, cur_lines
        if cur_key:
            val = "\n".join(cur_lines).strip()
            if not val:
                raise ValueError(f"Empty regex value for key '{cur_key}' in {path}")
            patterns[cur_key] = val
        cur_key, cur_lines = None, []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, val = line.split(":", 1)
            key = key.strip().upper()
            val = val.strip()
            if not key:
                raise ValueError(f"Invalid pattern line (empty key): {raw}")
            _flush()
            cur_key = key
            cur_lines = [val] if val else []
        else:
            if not cur_key:
                raise ValueError(f"Value line before any key in {path}: {raw}")
            cur_lines.append(line)
    _flush()
    return patterns


def _compile_patterns() -> Dict[str, re.Pattern]:
    compiled: Dict[str, re.Pattern] = {}
    compiled["HDR"] = re.compile(PATTERN_HDR, flags=re.MULTILINE | re.UNICODE)
    compiled["BULLET"] = re.compile(PATTERN_BULLET, flags=re.MULTILINE | re.UNICODE)
    return compiled


_PATTERNS = _compile_patterns()
HDR_RE = _PATTERNS["HDR"]
BULLET_RE = _PATTERNS["BULLET"]


def _sanitize_unicode(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace(NBSP, " ").replace(ZWSP, "").replace(BOM, "")
    s = SOFT_HYPHEN_RE.sub("", s)
    for bad, good in UNICODE_FIXES.items():
        s = s.replace(bad, good)
    s = CTRL_RE.sub("", s)
    return s


def normalize_text(
    s: str,
    *,
    profile: str = CHUNK_PROFILE,
    dehyphenate: bool = True,
    collapse_whitespace: bool = True,
) -> str:
    if not s:
        return ""
    s = _sanitize_unicode(s)
    s = s.replace("\r", "\n")
    if dehyphenate:
        s = HYPHEN_BREAK_RE.sub("", s)
    if collapse_whitespace:
        s = WS_RE.sub(" ", s)
    s = MULTINL_RE.sub("\n\n", s)
    s = "\n".join(line.strip() for line in s.split("\n"))
    if profile == "strict":
        s = re.sub(r"[ \t]+\n", "\n", s)
        s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def split_blocks(text: str) -> List[str]:
    lines = text.split("\n")
    blocks: List[str] = []
    buf: List[str] = []
    for ln in lines:
        if HDR_RE.match(ln) and buf:
            blocks.append("\n".join(buf).strip())
            buf = [ln]
        elif ln.strip() == "" and buf:
            blocks.append("\n".join(buf).strip())
            buf = []
        else:
            buf.append(ln)
    if buf:
        blocks.append("\n".join(buf).strip())
    return [b for b in blocks if b and len(b) >= 3]


def split_paragraphs(block: str) -> List[str]:
    paras: List[str] = []
    buf: List[str] = []
    for ln in block.split("\n"):
        if BULLET_RE.match(ln):
            if buf:
                paras.append("\n".join(buf).strip())
                buf = []
            paras.append(ln.strip())
        else:
            buf.append(ln)
    if buf:
        paras.append("\n".join(buf).strip())
    return [p for p in paras if p]


def approx_tokens(s: str) -> int:
    return max(1, len(s) // 4)


def smart_chunk(
    text: str,
    *,
    profile: str = CHUNK_PROFILE,
    max_tokens: int = CHUNK_MAX_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
    min_tokens: int = CHUNK_MIN_TOKENS,
    max_chars: Optional[int] = CHUNK_MAX_CHARS,
) -> List[str]:
    text = normalize_text(text, profile=profile)
    if not text:
        return []
    blocks = split_blocks(text) or [text]
    chunks: List[str] = []
    carry = ""
    buf: List[str] = []
    buf_toks = 0
    def _emit_buffer() -> Optional[str]:
        nonlocal carry, buf, buf_toks
        if not buf:
            return None
        chunk = ((carry + "\n") if carry else "") + "\n\n".join(buf).strip()
        if approx_tokens(chunk) >= min_tokens:
            if max_chars and len(chunk) > max_chars:
                return chunk[:max_chars].rstrip()
            chunks.append(chunk)
            carry = chunk[-int(overlap_tokens * 4):] if chunk else ""
        else:
            carry = chunk[-int(overlap_tokens * 4):] if chunk else carry
        buf, buf_toks = [], 0
        return chunks[-1] if chunks else None
    for blk in blocks:
        for p in split_paragraphs(blk):
            t = approx_tokens(p)
            if t >= max_tokens:
                _emit_buffer()
                if max_chars and len(p) > max_chars:
                    start = 0
                    while start < len(p):
                        end = min(len(p), start + max_chars)
                        part = p[start:end].strip()
                        if part:
                            chunks.append(part)
                            carry = part[-int(overlap_tokens * 4):]
                        start = end
                else:
                    chunks.append(p.strip())
                    carry = p[-int(overlap_tokens * 4):]
                continue
            if buf_toks + t > max_tokens:
                _emit_buffer()
                buf, buf_toks = [p], t
            else:
                buf.append(p)
                buf_toks += t
    _emit_buffer()
    out: List[str] = []
    seen: set[str] = set()
    for c in chunks:
        h = hashlib.sha1(c.encode("utf-8")).hexdigest()[:16]
        if h not in seen:
            seen.add(h)
            out.append(c)
    return out


