"""速度曲线：输入 4K/8K/16K/32K/64K/128K，输出固定 2K（在容器内运行）。

对比官方 2026-09-26 W3A4 基准（BetterBench）：
  prefill ≈ 4.1K tok/s，TTFT 2K=0.36s / 16K=2.88s / 64K=13.0s，decode 加权 184 tok/s。
"""
import json, time, urllib.request, uuid, os
from transformers import AutoTokenizer

TOK = AutoTokenizer.from_pretrained(os.environ.get("MODEL_DIR", "./model"))
U = "http://127.0.0.1:8080/v1/chat/completions"
BASE = ("设备维护与运行分析：本段记录用于评估长上下文吞吐，内容包含巡检、润滑、振动、温度、电流、"
        "绝缘、接地、联锁、备件、工单、交接班等要素，逐项说明状态与处置建议。")


def doc_of(n_tokens):
    block = BASE * (n_tokens // max(len(TOK.encode(BASE)), 1) + 3)
    return TOK.decode(TOK.encode(block)[:n_tokens])


def run(n):
    doc = doc_of(n)
    prompt = (f"[run {uuid.uuid4().hex[:8]}] 下面是附录资料。请基于它写一份完整的技术分析报告，"
              f"不少于 1800 字，分节展开，逐条讨论，不要照抄原文。\n\n附录：\n" + doc)
    body = json.dumps({
        "model": "R9700-Qwen3.8-27B-W3A4-HIP",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2048, "temperature": 0, "ignore_eos": True,
        "seed": 42, "stream": True, "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    t0 = time.time(); first = None; chars = 0; usage = None; finish = None
    req = urllib.request.Request(U, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            s = line.decode().strip()
            if not s.startswith("data: ") or s == "data: [DONE]":
                continue
            d = json.loads(s[6:])
            ch = d.get("choices") or []
            if ch and ch[0].get("delta", {}).get("content"):
                if first is None:
                    first = time.time()
                chars += len(ch[0]["delta"]["content"])
            if ch and ch[0].get("finish_reason"):
                finish = ch[0]["finish_reason"]
            if d.get("usage"):
                usage = d["usage"]
    end = time.time()
    pt = usage["prompt_tokens"]; ct = usage["completion_tokens"]
    ttft = first - t0
    dec = ct / max(end - first, 1e-9)
    pre = pt / max(ttft, 1e-9)
    print(json.dumps({"nominal": n, "prompt_tokens": pt, "completion_tokens": ct,
                      "ttft_s": round(ttft, 2), "prefill_tok_s": round(pre), "decode_tok_s": round(dec, 1),
                      "total_s": round(end - t0, 1), "finish": finish}), flush=True)


for n in (4096, 8192, 16384, 32768, 65536, 131072):
    run(n)
