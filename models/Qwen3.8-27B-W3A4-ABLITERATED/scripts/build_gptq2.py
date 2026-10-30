#!/usr/bin/env python3
"""GPTQ（分组块对角 Hessian）校准版 W3ROT 构建器。

显存约束：27B 4-bit + eager attention 只能跑 8x2048 = 16K token/批，
因此用 4 批（共 65,536 token）累积 Hessian；Hessian 按 128 列分组存块对角
（136 个 128x128），显存/内存都可控，量化时逐组做列级误差补偿。

用法：python3 build_gptq2.py --src /src --out /out --calib /calib/calib.txt
"""
import argparse, json, os, struct, time
import numpy as np
import torch
from transformers import (AutoTokenizer, Qwen3_5ForConditionalGeneration, BitsAndBytesConfig)

H128 = None
H128_NP = None
ROTATED = {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
           "linear_attn.in_proj_qkv", "linear_attn.in_proj_z",
           "mlp.gate_proj", "mlp.up_proj"}
GRID = (0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00)
GROUP_OF = {"self_attn.q_proj": "qkv", "self_attn.k_proj": "qkv", "self_attn.v_proj": "qkv",
            "linear_attn.in_proj_qkv": "gdn_in", "linear_attn.in_proj_z": "gdn_in",
            "mlp.gate_proj": "gateup", "mlp.up_proj": "gateup",
            "self_attn.o_proj": "o", "linear_attn.out_proj": "gdn_out", "mlp.down_proj": "down"}


def st_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def st_read(path, name):
    hdr, base = st_header(path)
    m = hdr[name]
    with open(path, "rb") as f:
        f.seek(base + m["data_offsets"][0])
        raw = f.read(m["data_offsets"][1] - m["data_offsets"][0])
    if m["dtype"] == "BF16":
        a = (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32)
    else:
        a = np.frombuffer(raw, np.float32)
    return a.reshape(m["shape"])


def bf16_bits(x):
    u = np.ascontiguousarray(x, np.float32).view(np.uint32)
    r = ((u >> 16) & 1) + 0x7FFF
    return ((u + r) >> 16).astype(np.uint16)


def st_write(path, tensors, meta):
    hdr = {"__metadata__": meta}
    off = 0
    blobs = []
    for name, dt, shp, b in tensors:
        hdr[name] = {"dtype": dt, "shape": list(shp), "data_offsets": [off, off + len(b)]}
        off += len(b)
        blobs.append(b)
    js = json.dumps(hdr, separators=(",", ":")).encode()
    js += b" " * ((-len(js)) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(js)))
        f.write(js)
        for b in blobs:
            f.write(b)


def pack_u(u):
    o, K = u.shape
    bits = np.zeros((o, 3 * K), np.uint8)
    for j in range(3):
        bits[:, j::3] = (u >> j) & 1
    return np.packbits(bits, axis=1, bitorder="little").tobytes()


def hadamard(n):
    H = np.array([[1.0]], np.float32)
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return (H / np.sqrt(n)).astype(np.float32)


def rot_w(W):
    n, K = W.shape
    return (W.reshape(n, K // 128, 128) @ H128).reshape(n, K)


def rot_x(x):
    N, K = x.shape
    return (x.reshape(N, K // 128, 128) @ H128).reshape(N, K)


def scale_search(g):
    """g: [n,1,128] -> s [n,1]（两个符号基准 x 7 档，取平方误差最小）"""
    pos = g.amax(2)
    neg = g.amin(2)
    s_pos = torch.maximum(pos / 3.0, -neg / 4.0)
    s_neg = -torch.maximum(pos / 4.0, -neg / 3.0)
    best_s = best_err = None
    for base in (s_pos, s_neg):
        for f in GRID:
            s = base * f
            s = torch.where(s.abs() < 1e-12, torch.full_like(s, 1e-12), s)
            u = torch.clamp(torch.round(g / s[:, :, None]) + 4, 0, 7)
            err = ((g - s[:, :, None] * (u - 4)) ** 2).sum(2)
            if best_err is None:
                best_err, best_s = err, s
            else:
                m = err < best_err
                best_err = torch.where(m, err, best_err)
                best_s = torch.where(m, s, best_s)
    return best_s


def group_gptq(W, Hb, block=32):
    """W [n,k] float32，Hb [G,128,128] -> u uint8 [n,k], s float32 [n,G]"""
    n, K = W.shape
    G = K // 128
    dev = W.device
    u_out = torch.empty((n, K), dtype=torch.uint8, device=dev)
    s_out = torch.empty((n, G), dtype=torch.float32, device=dev)
    for gi in range(G):
        c0 = gi * 128
        Wg = W[:, c0:c0 + 128].clone()
        Hg = Hb[gi].to(dev, non_blocking=False)
        damp = 0.01 * torch.diagonal(Hg).mean()
        Hg = Hg + torch.eye(128, device=dev) * damp
        Hinv = torch.cholesky_inverse(torch.linalg.cholesky(Hg))
        diag = torch.diagonal(Hinv).clone()
        s = scale_search(Wg.reshape(n, 1, 128)).reshape(n)
        s_out[:, gi] = s
        for b0 in range(0, 128, block):
            b1 = min(b0 + block, 128)
            Wb = Wg[:, b0:b1]
            u = torch.clamp(torch.round(Wb / s[:, None]) + 4, 0, 7)
            Wq = (u.float() - 4) * s[:, None]
            u_out[:, c0 + b0:c0 + b1] = u.to(torch.uint8)
            if b1 < 128:
                err = (Wb - Wq) / diag[b0:b1].unsqueeze(0)
                Wg[:, b1:] -= err @ Hinv[b0:b1, b1:]
        del Hg, Hinv
    return u_out, s_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/src")
    ap.add_argument("--out", default="/out")
    ap.add_argument("--calib", default="/calib/calib.txt")
    ap.add_argument("--batches", type=int, default=4)
    ap.add_argument("--nseq", type=int, default=8)
    ap.add_argument("--seqlen", type=int, default=2048)
    a = ap.parse_args()
    global H128, H128_NP
    os.makedirs(a.out, exist_ok=True)
    H128_NP = hadamard(128)
    H128 = torch.tensor(H128_NP, dtype=torch.float32, device="cuda")

    tok = AutoTokenizer.from_pretrained(a.src)
    text = open(a.calib, encoding="utf-8").read()
    ids = tok(text, add_special_tokens=False)["input_ids"]
    per = a.nseq * a.seqlen
    tot = per * a.batches
    if len(ids) < tot:
        raise SystemExit(f"calib too short {len(ids)} < {tot}")
    batches = [np.array(ids[i * per:(i + 1) * per], dtype=np.int64).reshape(a.nseq, a.seqlen)
               for i in range(a.batches)]

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    t0 = time.time()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        a.src, quantization_config=bnb, device_map={"": "cuda:0"}, dtype=torch.bfloat16,
        low_cpu_mem_usage=True, attn_implementation="eager")
    model.eval()
    for attr in ("lm_head",):
        if hasattr(model, attr):
            delattr(model, attr)
    try:
        del model.model.visual
    except Exception:
        pass
    torch.cuda.empty_cache()
    print(f"loaded {time.time()-t0:.0f}s free={torch.cuda.mem_get_info()[0]/2**30:.1f}GiB", flush=True)

    cfg = json.load(open(os.path.join(a.src, "config.json")))
    lt = cfg.get("layer_types") or cfg.get("text_config", {})["layer_types"]
    idx = json.load(open(os.path.join(a.src, "model.safetensors.index.json")))["weight_map"]
    mods = dict(model.named_modules())
    H = {}
    SEEN = set()
    t0 = time.time()

    def make_hook(li, site):
        gname = GROUP_OF[site]
        rotated = site in ROTATED

        def fn(module, inputs):
            x = inputs[0]
            if x.dim() == 3:
                x = x.reshape(-1, x.shape[-1])
            K = x.shape[-1]
            key = (li, gname)
            if key in SEEN:
                return
            SEEN.add(key)
            if key not in H:
                H[key] = torch.zeros((K // 128, 128, 128), dtype=torch.float32, device="cuda")
            h = H[key]
            for i in range(0, x.shape[0], 2048):
                c = x[i:i + 2048].float()
                if rotated:
                    c = rot_x(c)
                cg = c.reshape(-1, K // 128, 128)
                h += torch.einsum("ngk,ngl->gkl", cg, cg)
            del c, cg
        return fn

    for li in range(64):
        base = f"model.language_model.layers.{li}."
        sites = (["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                  "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"] if lt[li] == "full_attention" else
                 ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj",
                  "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"])
        for s in sites:
            mods[base + s].register_forward_pre_hook(make_hook(li, s))

    for bi, arr in enumerate(batches):
        SEEN.clear()
        ids_t = torch.from_numpy(arr).to("cuda")
        with torch.no_grad():
            model.model(input_ids=ids_t)
        del ids_t
        torch.cuda.empty_cache()
        print(f"pass {bi+1}/{a.batches} done {time.time()-t0:.0f}s free={torch.cuda.mem_get_info()[0]/2**30:.1f}GiB", flush=True)
    del model
    torch.cuda.empty_cache()

    wmap = {}
    for li in range(64):
        tensors = []
        base = f"model.language_model.layers.{li}."
        if lt[li] == "full_attention":
            sites = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                     "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
        else:
            sites = ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj",
                     "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
        for s in sites:
            W = torch.from_numpy(st_read(os.path.join(a.src, idx[base + s + ".weight"]),
                                         base + s + ".weight")).to("cuda")
            if s in ROTATED:
                W = rot_w(W)
            u, sc = group_gptq(W, H[(li, GROUP_OF[s])])
            del W
            u_np = u.to("cpu").numpy(); s_np = sc.to("cpu").numpy()
            del u, sc
            pref = base + s
            tensors.append((pref + ".weight_packed", "I32", (u_np.shape[0], 3 * u_np.shape[1] // 32), pack_u(u_np)))
            tensors.append((pref + ".weight_scale", "BF16", s_np.shape, bf16_bits(s_np).tobytes()))
            tensors.append((pref + ".weight_shape", "I64", (2,), np.array(u_np.shape, "<i8").tobytes()))
        if lt[li] != "full_attention":
            for s in ("linear_attn.in_proj_b", "linear_attn.in_proj_a"):
                Wn = st_read(os.path.join(a.src, idx[base + s + ".weight"]), base + s + ".weight")
                Wr = rot_w(torch.from_numpy(Wn).to("cuda"))
                tensors.append((base + s + ".weight_rot_h128", "BF16", tuple(Wr.shape),
                                bf16_bits(Wr.to("cpu").numpy()).tobytes()))
        fn = f"w3rot-L{li:02d}.safetensors"
        st_write(os.path.join(a.out, fn), tensors,
                 {"format": "paiton-w3rot-v1", "layer": str(li), "hadamard_block": "128"})
        for name, _, _, _ in tensors:
            wmap[name] = fn
        print(f"L{li:02d} {len(tensors)} tensors {time.time()-t0:.0f}s", flush=True)
    json.dump({"weight_map": wmap}, open(os.path.join(a.out, "w3rot-index.json"), "w"), indent=1)
    print("done", len(wmap), "tensors", f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
