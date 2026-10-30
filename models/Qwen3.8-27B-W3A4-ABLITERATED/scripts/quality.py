"""四个质量基准（对齐官方 2026-09-26 W3A4 README）：
  GSM8K 5-shot (1319) / HumanEval pass@1 (164) / MMLU-Pro 0-shot 14x100 / Needle @61,440 tok (80)

跑在 VM123 上，对 127.0.0.1:8080。结果写 JSONL，可断点续跑。
用法：python quality.py [gsm8k|humaneval|mmlu|needle|all] [并发]
"""
import json, os, random, re, subprocess, sys, tempfile, time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

URL = "http://127.0.0.1:8080/v1/chat/completions"
MODEL = "R9700-Qwen3.8-27B-W3A4-HIP"
OUT = os.environ.get("QUALITY_OUT", "./results")
NOCALC = ("本次是标准化评测。禁止调用 calc 工具，所有数值计算直接在回答中完成。")
LIMIT = int(os.environ.get("LIMIT", "0"))
MMLU_PER_CAT = int(os.environ.get("MMLU_PER_CAT", "100"))
MMLU_SEED = int(os.environ.get("MMLU_SEED", "42"))
NEEDLE_N = int(os.environ.get("NEEDLE_N", "80"))
os.makedirs(OUT, exist_ok=True)


def ask(messages, max_tokens=512, stop=None):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens,
            "temperature": 0, "seed": 42,
            "chat_template_kwargs": {"enable_thinking": False}}
    if stop:
        body["stop"] = stop
    for attempt in range(3):
        try:
            r = requests.post(URL, json=body, timeout=600)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"] or ""
        except Exception as e:
            if attempt == 2:
                return f"<ERR {e}>"
            time.sleep(2)


def load_done(path):
    done = {}
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            try:
                d = json.loads(line)
                done[d["id"]] = d
            except Exception:
                pass
    return done


def append(path, rec):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------- GSM8K 5-shot ----------------
def run_gsm8k(workers=8):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main")
    train = ds["train"].select(range(5))
    shots = "\n\n".join(f"问：{r['question']}\n答：{r['answer']}" for r in train)
    path = f"{OUT}/gsm8k.jsonl"
    done = load_done(path)
    items = [(i, r) for i, r in enumerate(ds["test"]) if str(i) not in done]
    if LIMIT: items = items[:LIMIT]
    print(f"gsm8k: {len(items)} 待跑 / 共 {len(ds['test'])}", flush=True)

    def one(i, r):
        gold = r["answer"].split("####")[-1].strip().replace(",", "")
        prompt = (f"{NOCALC}\n\n{shots}\n\n问：{r['question']}\n答：")
        txt = ask([{"role": "user", "content": prompt}], 512, stop=["\n问：", "\n\n问："])
        m = re.findall(r"####\s*\$?(-?[\d,]+(?:\.\d+)?)", txt)
        if not m:
            m = re.findall(r"(-?[\d,]+(?:\.\d+)?)\s*$", txt.strip())
        pred = m[-1].replace(",", "") if m else None
        ok = pred is not None and pred == gold
        rec = {"id": str(i), "gold": gold, "pred": pred, "ok": ok, "out": txt[-200:]}
        append(path, rec)
        return rec

    ok = 0
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(one, i, r) for i, r in items]
        for n, f in enumerate(as_completed(futs), 1):
            ok += f.result()["ok"]
            if n % 50 == 0:
                print(f"  {n}/{len(items)} ok={ok}", flush=True)
    allr = load_done(path)
    print(f"GSM8K: {sum(v['ok'] for v in allr.values())}/{len(allr)} = "
          f"{100*sum(v['ok'] for v in allr.values())/max(len(allr),1):.2f}%", flush=True)


# ---------------- HumanEval ----------------
CODE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.S)


def run_humaneval(workers=4):
    from datasets import load_dataset
    ds = load_dataset("openai/openai_humaneval", split="test")
    path = f"{OUT}/humaneval.jsonl"
    done = load_done(path)
    items = [r for r in ds if r["task_id"] not in done]
    if LIMIT: items = items[:LIMIT]
    print(f"humaneval: {len(items)} 待跑 / 共 {len(ds)}", flush=True)

    def one(r):
        prompt = (f"{NOCALC}\n\n补全下面的 Python 函数，只输出完整可运行的代码（含原函数签名），不要解释。\n\n"
                  f"```python\n{r['prompt']}```")
        txt = ask([{"role": "user", "content": prompt}], 1024)
        blocks = CODE_RE.findall(txt)
        code = blocks[0] if blocks else txt
        src = ("from typing import *\nimport math\nimport re\nimport collections\nimport itertools\n"
               "import heapq\nimport bisect\nimport string\n\n" + code + "\n\n" + r["test"] +
               f"\n\ncheck({r['entry_point']})\nprint('PASS')\n")
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(src)
            p = f.name
        try:
            res = subprocess.run([sys.executable, p], capture_output=True, text=True, timeout=15)
            ok = res.returncode == 0 and "PASS" in res.stdout
            err = (res.stderr.strip().splitlines() or [""])[-1][:150]
        except Exception as e:
            ok, err = False, str(e)[:150]
        finally:
            os.unlink(p)
        rec = {"id": r["task_id"], "ok": ok, "err": err}
        append(path, rec)
        return rec

    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(one, r) for r in items]
        for n, f in enumerate(as_completed(futs), 1):
            if n % 20 == 0:
                print(f"  {n}/{len(items)}", flush=True)
    allr = load_done(path)
    print(f"HumanEval: {sum(v['ok'] for v in allr.values())}/{len(allr)} = "
          f"{100*sum(v['ok'] for v in allr.values())/max(len(allr),1):.2f}%", flush=True)


# ---------------- MMLU-Pro 14 x 100 ----------------
def run_mmlu(workers=8):
    from datasets import load_dataset
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    by_cat = {}
    for r in ds:
        by_cat.setdefault(r["category"], []).append(r)
    cats = sorted(by_cat)
    import random as _rnd
    rng = _rnd.Random(MMLU_SEED)
    picks = []
    for c in cats:
        pool = by_cat[c]
        picks += rng.sample(pool, min(MMLU_PER_CAT, len(pool)))
    print(f"mmlu: {len(cats)} 类 x{MMLU_PER_CAT} (seed={MMLU_SEED}) = {len(picks)}", flush=True)
    path = f"{OUT}/mmlu.jsonl"
    done = load_done(path)
    items = [r for r in picks if r["question_id"] not in done]
    if LIMIT: items = items[:LIMIT]

    def one(r):
        opts = "\n".join(f"{chr(65+i)}. {o}" for i, o in enumerate(r["options"]))
        prompt = (f"{NOCALC}\n\n回答下面的单选题，只输出一行“答案：X”（X 是选项字母）。\n\n"
                  f"{r['question']}\n{opts}")
        txt = ask([{"role": "user", "content": prompt}], 64)
        m = re.search(r"答案[:：]\s*([A-J])", txt) or re.search(r"\b([A-J])\b", txt)
        pred = m.group(1) if m else None
        ok = pred == r["answer"]
        rec = {"id": r["question_id"], "cat": r["category"], "gold": r["answer"], "pred": pred, "ok": ok}
        append(path, rec)
        return rec

    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(one, r) for r in items]
        for n, f in enumerate(as_completed(futs), 1):
            if n % 100 == 0:
                print(f"  {n}/{len(items)}", flush=True)
    allr = load_done(path)
    tot = sum(v["ok"] for v in allr.values())
    print(f"MMLU-Pro: {tot}/{len(allr)} = {100*tot/max(len(allr),1):.2f}%", flush=True)
    for c in cats:
        sub = [v for v in allr.values() if v.get("cat") == c]
        if sub:
            print(f"   {c}: {sum(x['ok'] for x in sub)}/{len(sub)}", flush=True)


# ---------------- Needle @ 61,440 ----------------
def run_needle(workers=2):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.environ.get("TOK_DIR", "./model"))
    path = f"{OUT}/needle.jsonl"
    done = load_done(path)
    base = ("设备维护记录：巡检、润滑、振动、温度、电流、绝缘、接地、联锁、备件、工单、交接班等要素，"
            "逐项记录状态与处置建议，确保设备长期稳定运行。")
    ids = tok.encode(base * 400)

    def make_doc(rng):
        code = f"{rng.randint(1000000, 9999999)}"
        total = 61440
        pos = rng.randint(30000, 31000)
        doc = tok.decode(ids[:total])
        needle = f"【记录】备用服务器的 SSH 端口是 {code}。"
        doc = doc[: len(doc) // 2] + needle + doc[len(doc) // 2:]
        return doc, code

    def one(i):
        rng = random.Random(1000 + i)
        doc, code = make_doc(rng)
        prompt = (f"以下是设备维护手册。请回答：备用服务器的 SSH 端口是多少？只回答数字。\n\n{doc}")
        txt = ask([{"role": "user", "content": prompt}], 32)
        ok = code in txt
        rec = {"id": str(i), "code": code, "ok": ok, "out": txt[:60]}
        append(path, rec)
        return rec

    items = [i for i in range(NEEDLE_N) if str(i) not in done]
    if LIMIT: items = items[:LIMIT]
    print(f"needle: {len(items)} 待跑", flush=True)
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(one, i) for i in items]
        for n, f in enumerate(as_completed(futs), 1):
            if n % 10 == 0:
                print(f"  {n}/{len(items)}", flush=True)
    allr = load_done(path)
    print(f"Needle: {sum(v['ok'] for v in allr.values())}/{len(allr)}", flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    w = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    if which in ("gsm8k", "all"):
        run_gsm8k(w)
    if which in ("humaneval", "all"):
        run_humaneval(max(2, w // 2))
    if which in ("mmlu", "all"):
        run_mmlu(w)
    if which in ("needle", "all"):
        run_needle(2)
