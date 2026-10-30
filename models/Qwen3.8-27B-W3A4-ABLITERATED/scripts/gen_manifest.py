#!/usr/bin/env python3
"""为自建 paiton-w3rot 覆盖层生成 manifest 与 SHA256SUMS。

做法：克隆官方 manifest 的结构（loader 可能依赖它的键），
只替换可验证的事实字段：分片哈希/字节数、张量数、创建时间、来源与量化方法。
"""
import json, os, sys, hashlib, datetime

def sha256(p, buf=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def main(out_dir, official_manifest, src_note, src_dir=None):
    man = json.load(open(official_manifest))
    files = {}
    sums = []
    for i in range(64):
        fn = f"w3rot-L{i:02d}.safetensors"
        p = os.path.join(out_dir, fn)
        s = sha256(p)
        files[fn] = {"sha256": s, "bytes": os.path.getsize(p)}
        sums.append((s, fn))
    man["files"] = files
    man["tensors"] = 1296
    man["created"] = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    import os as _os
    man["weights_version"] = _os.environ.get("W3ROT_VERSION", "2026-10-29-abl-rtn-c1")
    man.pop("finetune", None)
    man.pop("previous_finetune", None)
    import os as _os
    man["gptq"] = json.loads(_os.environ["W3ROT_GPTQ_JSON"]) if _os.environ.get("W3ROT_GPTQ_JSON") else {
        "method": "rtn + per-group MSE scale search (no calibration data)",
        "group": 128,
        "scale_candidates": 7,
        "notes": "candidate scales derived analytically per (row,128) group for both scale signs; "
                 "lowest squared error kept. No Hessian/GPTQ, no scale distillation.",
    }
    man["derived_from"] = src_note
    # side 来源哈希 = sha256(源 bf16 的 in_proj_b 字节 ‖ in_proj_a 字节)，逐层
    if src_dir and "side" in man:
        idx = json.load(open(os.path.join(src_dir, "model.safetensors.index.json")))["weight_map"]
        cfg = json.load(open(os.path.join(src_dir, "config.json")))
        lt = cfg.get("layer_types") or cfg.get("text_config", {})["layer_types"]
        def raw_bytes(name):
            p = os.path.join(src_dir, idx[name])
            with open(p, "rb") as f:
                n = int.from_bytes(f.read(8), "little")
                hj = json.loads(f.read(n))
                base = 8 + n
                m = hj[name]
                f.seek(base + m["data_offsets"][0])
                return f.read(m["data_offsets"][1] - m["data_offsets"][0])
        by_layer = {}
        for li in range(64):
            if lt[li] == "full_attention":
                continue
            rb = raw_bytes(f"model.language_model.layers.{li}.linear_attn.in_proj_b.weight")
            ra = raw_bytes(f"model.language_model.layers.{li}.linear_attn.in_proj_a.weight")
            by_layer[str(li)] = hashlib.sha256(rb + ra).hexdigest()
        man["side"]["gdn_in_proj"]["source_sha256_by_layer"] = by_layer
    with open(os.path.join(out_dir, "w3rot-manifest.json"), "w") as f:
        json.dump(man, f, indent=1)
    with open(os.path.join(out_dir, "SHA256SUMS"), "w") as f:
        for s, fn in sums:
            f.write(f"{s}  {fn}\n")
        ms = sha256(os.path.join(out_dir, "w3rot-manifest.json"))
        f.write(f"{ms}  w3rot-manifest.json\n")
    print("manifest sha256:", sha256(os.path.join(out_dir, "w3rot-manifest.json")))
    print("files:", len(files), "tensors:", man["tensors"])

if __name__ == "__main__":
    note = {
        "base_checkpoint": "unsloth/Qwen3.8-27B-NVFP4@f0b7c9e722f5565102fff8481c99e4d86ae099c7",
        "source_weights": "huihui-ai/Huihui-Qwen3.8-27B-abliterated@739e3c5b89849f6c238ce1e5b70008612ae42cdd",
        "produced_by": "make_w3rot.py (RTN + per-group MSE scale search, Hadamard-128 rotation)",
        "note": "非官方权重；仅 attn/GDN/MLP 投影来自 abliterated 检查点，其余张量仍取自基座。",
    }
    main(sys.argv[1], sys.argv[2], note, sys.argv[3] if len(sys.argv)>3 else None)
