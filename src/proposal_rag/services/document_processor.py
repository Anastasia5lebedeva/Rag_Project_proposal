import os, re, json, argparse, hashlib
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parents[3] / "resources" / "prompts"
WS_RE = re.compile(r"[ \t\u00A0]+")
MULTINL_RE = re.compile(r"\n{3,}")
HDR_RE = re.compile((PROMPTS_DIR / "hdr_pattern.txt").read_text(encoding="utf-8").strip())
BULLET_RE = re.compile((PROMPTS_DIR / "bullet_pattern.txt").read_text(encoding="utf-8").strip())
















def approx_tokens(s: str) -> int:
    return max(1, len(s) // 4)


def normalize_text(s: str) -> str:
    s = s.replace("\r", "\n")
    s = WS_RE.sub(" ", s)
    s = MULTINL_RE.sub("\n\n", s)
    s = "\n".join(line.strip() for line in s.split("\n"))
    return s.strip()


def split_blocks(text: str) -> list[str]:
    lines = text.split("\n")
    blocks, buf = [], []
    for ln in lines:
        if HDR_RE.match(ln) and buf:
            blocks.append("\n".join(buf).strip()); buf = [ln]
        elif ln.strip() == "" and buf:
            blocks.append("\n".join(buf).strip()); buf = []
        else:
            buf.append(ln)
    if buf: blocks.append("\n".join(buf).strip())
    return [b for b in blocks if b and len(b) >= 3]

def split_paragraphs(block: str) -> list[str]:
    paras, buf = [], []
    for ln in block.split("\n"):
        if BULLET_RE.match(ln):
            if buf: paras.append("\n".join(buf).strip()); buf = []
            buf.append(ln)
        else:
            buf.append(ln)
    if buf: paras.append("\n".join(buf).strip())
    return [p for p in paras if p]

def smart_chunk(text: str, max_tokens: int = 1600, overlap_tokens: int = 250, min_tokens: int = 150) -> list[str]:
    text = normalize_text(text)
    if not text: return []
    blocks = split_blocks(text) or [text]
    chunks, carry = [], ""
    buf, buf_toks = [], 0

    for blk in blocks:
        for p in split_paragraphs(blk):
            t = approx_tokens(p)
            if t >= max_tokens:
                if buf:
                    chunk = ((carry + "\n") if carry else "") + "\n\n".join(buf).strip()
                    if approx_tokens(chunk) >= min_tokens: chunks.append(chunk.strip())
                    carry = chunk[-int(overlap_tokens*4):] if chunk else ""
                    buf, buf_toks = [], 0
                chunks.append(p.strip())
                carry = p[-int(overlap_tokens*4):]
                continue
            if buf_toks + t > max_tokens:
                chunk = ((carry + "\n") if carry else "") + "\n\n".join(buf).strip()
                if approx_tokens(chunk) >= min_tokens: chunks.append(chunk.strip())
                carry = chunk[-int(overlap_tokens*4):] if chunk else ""
                buf, buf_toks = [p], t
            else:
                buf.append(p); buf_toks += t

    if buf:
        chunk = ((carry + "\n") if carry else "") + "\n\n".join(buf).strip()
        if approx_tokens(chunk) >= min_tokens or not chunks:
            chunks.append(chunk.strip())
    dedup, seen = [], set()
    for c in chunks:
        k = c[:200]
        if k in seen: continue
        seen.add(k); dedup.append(c)
    return dedup
