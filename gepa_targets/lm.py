"""DSPy LM that runs Claude through the local `claude -p` CLI.

This uses the Claude subscription the CLI is logged in with, so no API key or credits are needed.

    from gepa_targets.lm import claude_lm
    student = claude_lm("haiku"); reflection = claude_lm("sonnet")

Guardrails (so a run can't burn through the plan's usage limits), all overridable by env var:
  LLM_ALLOWED_MODELS    models allowed at all            (default "haiku,sonnet"; add "opus" explicitly)
  LLM_MAX_CONCURRENT    simultaneous claude -p processes (default 6)
  LLM_MAX_CALLS         calls per Python process         (default 2000)
  LLM_MAX_INPUT_TOKENS  input tokens per Python process  (default 10,000,000)
  LLM_CALL_LOG          JSONL log of every call          (default results/llm_calls.jsonl)
Once a cap is hit, every further call raises BudgetExceeded immediately without calling Claude.
DSPy cache hits never reach the engine, so they cost nothing and don't count.
"""

import json
import os
import pathlib
import subprocess
import tempfile
import threading
import time

import dspy
from dspy.clients.engines.base import validate_request
from dspy.lm15 import Message, Response, TextPart, Usage

ALLOWED_MODELS = {m.strip() for m in os.environ.get("LLM_ALLOWED_MODELS", "haiku,sonnet").split(",") if m.strip()}
MAX_CONCURRENT = int(os.environ.get("LLM_MAX_CONCURRENT", 6))
MAX_CALLS = int(os.environ.get("LLM_MAX_CALLS", 2000))
MAX_INPUT_TOKENS = int(os.environ.get("LLM_MAX_INPUT_TOKENS", 10_000_000))
CALL_LOG = pathlib.Path(os.environ.get("LLM_CALL_LOG", "results/llm_calls.jsonl"))

_EMPTY_CWD = tempfile.mkdtemp(prefix="claude_cli_lm_")  # so no CLAUDE.md or project settings leak in
_SLOTS = threading.BoundedSemaphore(MAX_CONCURRENT)
_LOCK = threading.Lock()
_USAGE = {"calls": 0, "in": 0, "out": 0, "by_model": {}}


class BudgetExceeded(RuntimeError):
    pass


def usage():
    """Claude calls and tokens used so far in this Python process."""
    with _LOCK:
        return json.loads(json.dumps(_USAGE))


def _text(parts):
    return "".join(p.text for p in parts if isinstance(p, TextPart))


class ClaudeCLIEngine:
    def __init__(self, model="haiku", timeout=300):
        if model not in ALLOWED_MODELS:
            raise ValueError(f"model '{model}' is not in LLM_ALLOWED_MODELS={sorted(ALLOWED_MODELS)}; "
                             "add it explicitly if you really want it")
        self.model = model
        self.timeout = timeout

    def run_text(self, prompt, system=None):
        """One headless call. Returns (text, {'in': input tokens, 'out': output tokens})."""
        with _LOCK:
            if _USAGE["calls"] >= MAX_CALLS or _USAGE["in"] >= MAX_INPUT_TOKENS:
                raise BudgetExceeded(f"budget reached: {_USAGE['calls']} calls, {_USAGE['in']:,} input tokens "
                                     f"(caps {MAX_CALLS} calls / {MAX_INPUT_TOKENS:,} tokens)")
            _USAGE["calls"] += 1
        cmd = ["claude", "-p", "--model", self.model, "--tools", "", "--strict-mcp-config",
               "--no-session-persistence", "--output-format", "json",
               "--system-prompt", system or "You are a helpful assistant."]
        t = time.time()
        last = None
        for attempt in range(4):  # transient empty failures (throttling) get retried with backoff
            with _SLOTS:
                r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=self.timeout,
                                   cwd=_EMPTY_CWD)
            try:
                j = json.loads(r.stdout)
            except json.JSONDecodeError:
                j = {"is_error": True, "result": (r.stderr or r.stdout)[:300]}
            if r.returncode == 0 and not j.get("is_error"):
                break
            last = f"rc={r.returncode} subtype={j.get('subtype')} result={str(j.get('result'))[:200]}"
            with _LOCK:
                CALL_LOG.parent.mkdir(parents=True, exist_ok=True)
                with CALL_LOG.open("a") as f:
                    f.write(json.dumps({"t": round(time.time()), "model": self.model, "error": last,
                                        "attempt": attempt}) + "\n")
            time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError(f"claude -p failed after retries: {last}")
        u = j.get("usage", {})
        tin = u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0) + u.get("cache_creation_input_tokens", 0)
        tout = u.get("output_tokens", 0)
        with _LOCK:
            _USAGE["in"] += tin
            _USAGE["out"] += tout
            m = _USAGE["by_model"].setdefault(self.model, {"calls": 0, "in": 0, "out": 0})
            m["calls"] += 1
            m["in"] += tin
            m["out"] += tout
            CALL_LOG.parent.mkdir(parents=True, exist_ok=True)
            with CALL_LOG.open("a") as f:
                f.write(json.dumps({"t": round(t), "model": self.model, "in": tin, "out": tout,
                                    "seconds": round(time.time() - t, 1)}) + "\n")
        return j.get("result", ""), {"in": tin, "out": tout}

    def complete(self, request):
        validate_request(request)
        system = None
        if request.system is not None:
            system = request.system if isinstance(request.system, str) else _text(request.system)
        msgs = [(m.role, _text(m.parts)) for m in request.messages]
        if len(msgs) == 1:
            prompt = msgs[0][1]
        else:  # few-shot or multi-turn: flatten into one transcript
            prompt = "\n\n".join(f"<{role}>\n{text}\n</{role}>" for role, text in msgs) + "\n\n<assistant>\n"
        text, u = self.run_text(prompt, system)
        return Response(id=None, model=f"claude-cli/{self.model}",
                        message=Message.assistant([TextPart(text)]), finish_reason="stop",
                        usage=Usage(input_tokens=u["in"], output_tokens=u["out"], total_tokens=u["in"] + u["out"]))

    def close(self):
        pass


def claude_lm(model="haiku", **kwargs):
    return dspy.BaseLM(model=f"claude-cli/{model}", engine=ClaudeCLIEngine(model), **kwargs)
