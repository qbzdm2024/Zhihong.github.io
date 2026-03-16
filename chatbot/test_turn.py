#!/usr/bin/env python3
"""
Single-turn prompt tester for the HF Chatbot.

Usage:
    python test_turn.py                        # interactive prompt
    python test_turn.py "I have swollen ankles" --mode triage
    python test_turn.py "What foods should I avoid?" --rag-k 4

Requirements:
    pip install openai
"""

import argparse
import json
import os
import sys
import textwrap
import time

# ── Dependency check ──────────────────────────────────────────────────────────
try:
    from openai import OpenAI
except ImportError:
    sys.exit("openai package not found. Run: pip install openai")

# ── System prompt (mirrors chat-engine.js buildSystemPrompt) ─────────────────

BASE_PROMPT = """\
You are a compassionate, knowledgeable heart failure self-management assistant.
Your purpose is to educate heart failure patients and caregivers about heart failure management only.

## Scope Boundary
You ONLY answer questions related to heart failure and its management (symptoms, diet, exercise, medications, monitoring, triage).
If a question is clearly outside this scope, respond with:
"I'm here to support you with heart failure-related questions. For [topic], you may want to consult [appropriate resource]."

## Citation Rules — CRITICAL
- Use INLINE citations: write "According to [Source Name](URL), ..."
- End your response with a "## References" section listing every source cited.
- ONLY cite a source if it DIRECTLY supports the specific claim being made.
- If the knowledge base does NOT contain clear evidence for a claim, say so explicitly.
- Never cite a source about Topic A to support a claim about Topic B.

## Core Principles
1. Cite sources INLINE.
2. Be compassionate but clear. Use plain language.
3. Safety first — when in doubt, recommend contacting the healthcare team.
4. Never replace medical advice.

## Knowledge Base Context
{context}

## Response Format
- Weave inline citations naturally into sentences
- Use headers and bullets for clarity
- End with a ## References section"""

MODE_SUFFIXES = {
    "education": """

## EDUCATION MODE
The patient is asking a general heart failure question, not reporting current symptoms.
Do NOT provide a triage zone. Provide concise, evidence-based educational information.""",

    "triage": """

## THIS MESSAGE CONTAINS CURRENT SYMPTOM DESCRIPTIONS
The patient is describing symptoms they are experiencing right now.
- Acknowledge the specific symptoms with empathy.
- Provide relevant education with inline citations.
- Note that a triage assessment is displayed below your response.
- Do NOT repeat zone classifications in the text — the triage card shows them.""",

    "cardiac": """

## EDUCATION MODE — CARDIAC QUERY / POSSIBLE HF CONTEXT
The patient described a cardiac symptom (e.g. fast heartbeat, palpitations, irregular rhythm)
but has NOT mentioned having heart failure.

Instructions:
1. Briefly answer their question in the context of heart failure.
2. After your answer, ask ONE clarifying question:
   "Do you have a diagnosis of heart failure or are you being followed by a cardiologist?"
3. Do NOT provide a triage zone in this response.""",
}


def build_system_prompt(context: str, mode: str) -> str:
    suffix = MODE_SUFFIXES.get(mode, MODE_SUFFIXES["education"])
    return BASE_PROMPT.format(context=context) + suffix


# ── Simple keyword RAG (no embeddings — mirrors JS retrieve()) ────────────────

def load_knowledge_base(kb_path: str) -> list[dict]:
    try:
        with open(kb_path, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("chunks", [])
    except FileNotFoundError:
        print(f"[warn] Knowledge base not found at {kb_path} — running without RAG context.")
        return []
    except Exception as e:
        print(f"[warn] Failed to load KB: {e} — running without RAG context.")
        return []


def keyword_score(chunk: dict, query: str) -> float:
    """Simple keyword overlap score (mirrors JS TF-IDF-lite logic)."""
    words = set(query.lower().split())
    text  = (chunk.get("text", "") + " " + " ".join(chunk.get("tags", []))).lower()
    hits  = sum(1 for w in words if len(w) > 3 and w in text)
    return hits / max(len(words), 1)


def retrieve(chunks: list[dict], query: str, k: int = 6) -> list[dict]:
    scored = [(keyword_score(c, query), c) for c in chunks]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:k]]


def build_context(chunks: list[dict]) -> str:
    if not chunks:
        return "No relevant knowledge base content found."
    parts = []
    for i, c in enumerate(chunks, 1):
        src = c.get("source", {}).get("name", c.get("sourceId", "Unknown"))
        url = c.get("source", {}).get("url", "")
        ref = f"[{src}]({url})" if url else src
        parts.append(f"[{i}] Source: {ref}\n{c.get('text', '').strip()}")
    return "\n\n---\n\n".join(parts)


# ── Auto-detect mode ──────────────────────────────────────────────────────────

TRIAGE_PATTERNS = [
    r"(i (am|feel|have|notice|been|keep|start|started))",
    r"(my (ankle|leg|weight|breath|chest|heart|stomach|head))",
    r"(swollen|swelling|gained|can.t breathe|short of breath|chest pain|dizzy|confused|fainting)",
]

CARDIAC_KW = r"(heart.?racing|palpitation|irregular.*beat|rapid pulse|fast heart)"

import re

def auto_detect_mode(message: str) -> str:
    msg = message.lower()
    is_personal = any(re.search(p, msg) for p in TRIAGE_PATTERNS)
    triage_symptoms = any(kw in msg for kw in [
        "breath", "chest", "fatigue", "tired", "weight", "confused",
        "swell", "dizzy", "lightheaded", "faint"
    ])
    if is_personal and triage_symptoms:
        return "triage"
    if re.search(CARDIAC_KW, msg, re.I) and not triage_symptoms:
        return "cardiac"
    return "education"


# ── Pretty print ──────────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BLUE   = "\033[94m"

def divider(title: str = "", char: str = "─", width: int = 72, color: str = DIM) -> None:
    if title:
        pad = max(0, width - len(title) - 2)
        print(f"{color}{char * 2} {BOLD}{title}{RESET}{color} {char * pad}{RESET}")
    else:
        print(f"{color}{char * width}{RESET}")


def wrap_print(text: str, indent: int = 2, width: int = 70, color: str = RESET) -> None:
    for line in text.splitlines():
        if line.strip():
            wrapped = textwrap.fill(line, width=width, subsequent_indent=" " * indent)
            print(f"{color}  {wrapped}{RESET}")
        else:
            print()


# ── Main ──────────────────────────────────────────────────────────────────────

def run_turn(
    message: str,
    api_key: str,
    model: str = "gpt-4o-mini",
    mode: str = "auto",
    rag_k: int = 6,
    kb_path: str = None,
    history: list[dict] = None,
    show_system_prompt: bool = True,
    show_chunks: bool = True,
) -> dict:

    # ── Resolve KB path ───────────────────────────────────────────────────────
    if kb_path is None:
        kb_path = os.path.join(os.path.dirname(__file__), "data", "hf-knowledge.json")

    # ── Load KB and retrieve ──────────────────────────────────────────────────
    chunks_all = load_knowledge_base(kb_path)
    retrieved  = retrieve(chunks_all, message, rag_k)
    context    = build_context(retrieved)

    # ── Resolve mode ──────────────────────────────────────────────────────────
    resolved_mode = auto_detect_mode(message) if mode == "auto" else mode

    # ── Build system prompt ───────────────────────────────────────────────────
    system_prompt = build_system_prompt(context, resolved_mode)

    # ── Build messages array ──────────────────────────────────────────────────
    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": message})

    # ── Print config ──────────────────────────────────────────────────────────
    print()
    divider("SINGLE-TURN TEST", "═", color=CYAN)
    print(f"  {BOLD}Model:{RESET}  {model}")
    print(f"  {BOLD}Mode:{RESET}   {resolved_mode}  {DIM}(input: {mode}){RESET}")
    print(f"  {BOLD}RAG k:{RESET}  {rag_k}  →  {len(retrieved)} chunks retrieved")
    print(f"  {BOLD}Hist:{RESET}   {len(history or [])} prior message(s)")

    # ── User message ──────────────────────────────────────────────────────────
    print()
    divider("USER MESSAGE", color=YELLOW)
    wrap_print(message, color=YELLOW)

    # ── Retrieved chunks ──────────────────────────────────────────────────────
    if show_chunks and retrieved:
        print()
        divider(f"RETRIEVED CHUNKS ({len(retrieved)})", color=BLUE)
        for i, c in enumerate(retrieved, 1):
            src = c.get("source", {}).get("name", c.get("sourceId", "?"))
            tags = ", ".join(c.get("tags", []))
            print(f"  {DIM}[{i}] {src}  |  tags: {tags}{RESET}")
            snippet = c.get("text", "")[:160].replace("\n", " ")
            print(f"      {snippet}…")

    # ── System prompt ─────────────────────────────────────────────────────────
    if show_system_prompt:
        print()
        divider(f"SYSTEM PROMPT ({len(system_prompt)} chars)", color=DIM)
        for line in system_prompt.splitlines():
            print(f"  {DIM}{line}{RESET}")

    # ── Call API ──────────────────────────────────────────────────────────────
    print()
    divider("CALLING API…", color=DIM)
    client    = OpenAI(api_key=api_key)
    t0        = time.time()

    response  = client.chat.completions.create(
        model=model,
        max_tokens=1500,
        messages=messages,
    )
    elapsed   = time.time() - t0
    content   = response.choices[0].message.content or ""
    usage     = response.usage

    # ── Response ──────────────────────────────────────────────────────────────
    print()
    divider("ASSISTANT RESPONSE", color=GREEN)
    wrap_print(content, color=GREEN)

    # ── Usage ─────────────────────────────────────────────────────────────────
    print()
    divider("USAGE", color=DIM)
    print(f"  Prompt tokens:     {usage.prompt_tokens}")
    print(f"  Completion tokens: {usage.completion_tokens}")
    print(f"  Total tokens:      {usage.total_tokens}")
    print(f"  Elapsed:           {elapsed:.2f}s")
    print(f"  Finish reason:     {response.choices[0].finish_reason}")
    divider("", "═", color=CYAN)
    print()

    return {
        "content": content,
        "usage": {"prompt": usage.prompt_tokens, "completion": usage.completion_tokens, "total": usage.total_tokens},
        "elapsed": elapsed,
        "mode": resolved_mode,
        "retrieved_chunks": len(retrieved),
        "system_prompt": system_prompt,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Test a single conversation turn against the HF chatbot system prompt."
    )
    parser.add_argument("message", nargs="?", help="User message (prompted if omitted)")
    parser.add_argument("--model",   default="gpt-4o-mini",
                        choices=["gpt-4o-mini", "gpt-4o", "o1-mini", "o1"])
    parser.add_argument("--mode",    default="auto",
                        choices=["auto", "education", "triage", "cardiac"],
                        help="System prompt mode (default: auto-detect)")
    parser.add_argument("--rag-k",   type=int, default=6,
                        help="Number of KB chunks to retrieve (default: 6)")
    parser.add_argument("--no-chunks",       action="store_true",
                        help="Hide retrieved chunks")
    parser.add_argument("--no-system-prompt", action="store_true",
                        help="Hide system prompt output")
    parser.add_argument("--kb",      default=None,
                        help="Path to hf-knowledge.json (auto-detected if omitted)")
    args = parser.parse_args()

    # API key: CLI env var > OPENAI_API_KEY env var > prompt
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        api_key = input("OpenAI API key (sk-...): ").strip()
    if not api_key:
        sys.exit("No API key provided.")

    # Message
    message = args.message
    if not message:
        print("Enter the user message (Ctrl+D or empty line to finish):")
        lines = []
        try:
            while True:
                line = input()
                if not line:
                    break
                lines.append(line)
        except EOFError:
            pass
        message = "\n".join(lines).strip()
    if not message:
        sys.exit("No message provided.")

    run_turn(
        message=message,
        api_key=api_key,
        model=args.model,
        mode=args.mode,
        rag_k=args.rag_k,
        kb_path=args.kb,
        show_chunks=not args.no_chunks,
        show_system_prompt=not args.no_system_prompt,
    )


if __name__ == "__main__":
    main()
