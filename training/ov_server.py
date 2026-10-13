#!/usr/bin/env python3
"""OpenAI-compatible shim around OpenVINO GenAI so aiproof's "llama.cpp
server" provider preset can talk to a model on the Intel NPU, iGPU or CPU
with no daemon changes.

  training/ov_server.py --model-dir ~/.local/share/aiproof/models/aiproof-edit-1.5b-ov --device NPU

The ChatML prompt is built here, byte-identical to training, rather than by
the tokenizer's template. One request at a time (the NPU pipeline is not
re-entrant); first NPU compile takes a few minutes and is cached.
"""

import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def chatml(messages: list) -> str:
    parts = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages]
    return "".join(parts) + "<|im_start|>assistant\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--device", default=os.environ.get("AIPROOF_OV_DEVICE", "NPU"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--max-prompt-len", type=int, default=1024)
    ap.add_argument("--min-response-len", type=int, default=128)
    args = ap.parse_args()

    import openvino_genai as ov_genai

    props = {"CACHE_DIR": os.path.expanduser("~/.cache/aiproof-ov")}
    if args.device.upper() == "NPU":
        props.update({"MAX_PROMPT_LEN": args.max_prompt_len,
                      "MIN_RESPONSE_LEN": args.min_response_len,
                      "GENERATE_HINT": "BEST_PERF"})
    t0 = time.monotonic()
    pipe = ov_genai.LLMPipeline(args.model_dir, args.device.upper(), **props)
    print(f"loaded {args.model_dir} on {args.device} in {time.monotonic() - t0:.1f}s", flush=True)
    lock = threading.Lock()
    model_name = os.path.basename(os.path.normpath(args.model_dir))

    class Handler(BaseHTTPRequestHandler):
        def _json(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            if self.path.startswith("/v1/models"):
                return self._json(200, {"object": "list", "data": [{"id": model_name, "object": "model"}]})
            return self._json(200, {"status": "ok", "device": args.device})

        def do_POST(self):  # noqa: N802
            if not self.path.startswith("/v1/chat/completions"):
                return self._json(404, {"error": {"message": "not found"}})
            length = int(self.headers.get("Content-Length", "0"))
            try:
                req = json.loads(self.rfile.read(length) or b"{}")
                messages = req["messages"]
            except (ValueError, KeyError):
                return self._json(400, {"error": {"message": "bad request"}})
            cfg = ov_genai.GenerationConfig()
            cfg.max_new_tokens = int(req.get("max_tokens") or 256)
            cfg.do_sample = False
            prompt = chatml(messages)
            t0 = time.monotonic()
            with lock:
                text = pipe.generate(prompt, cfg, apply_chat_template=False)
            text = str(text).split("<|im_end|>")[0].strip()
            self._json(200, {
                "id": "ov-" + str(int(t0)), "object": "chat.completion", "model": model_name,
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": text}}],
                "aiproof_timing_s": round(time.monotonic() - t0, 3),
            })

        def log_message(self, fmt, *a):  # quieter
            sys.stderr.write("%s %s\n" % (self.log_date_time_string(), fmt % a))

    print(f"serving http://{args.host}:{args.port}/v1/chat/completions", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
