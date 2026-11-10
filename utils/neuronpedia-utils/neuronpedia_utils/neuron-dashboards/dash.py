"""Single-pass neuron dashboard data for every MLP neuron of one model.

A neuron's activation is a = act_fn(gate) * up, the input to down_proj. "Gate open" means gate > 0.
Position 0 of each document (BOS for Gemma) is never counted. For each neuron we keep:
  top / bot      K_MAIN gate-open examples with the largest and the most negative a (one per document)
  etop / ebot    K_ECHO gate-closed examples ("echoes") with the largest and the most negative a
  band0..band6   K_BAND random examples per activation band (band edges from a calibration histogram)
  hist, stats    sigma-relative log histogram split by gate state, and sums, on subsampled tokens
For each example: doc id, peak position, a, gate and up at the peak, the activations at offsets
WIN_LO..WIN_HI around the peak (NaN outside the document), and one gate-closed bit per window token.

Stages: sigma (per-neuron RMS) -> calib (full histogram -> band edges) -> main -> logits.
The main stage writes a checkpoint now and then and resumes from it.
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {
    "qwen3-32b": "Qwen/Qwen3-32B",
    "gemma-4-31b-it": "google/gemma-4-31B-it",
    "llama3.3-70b-it": "meta-llama/Llama-3.3-70B-Instruct",
    "qwen3-8b": "Qwen/Qwen3-8B",
    "qwen3.5-4b": "Qwen/Qwen3.5-4B",
    "gemma-4-12b-it": "google/gemma-4-12B-it",
    "gemma-4-12b": "google/gemma-4-12B",
    "llama3.1-8b-it": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen3-14b": "Qwen/Qwen3-14B",
}
K_MAIN, K_ECHO, K_BAND = 24, 8, 5
WIN_LO, WIN_HI = -47, 16
W = WIN_HI - WIN_LO + 1
# Band edges are quantiles of a; band i is [Q[i], Q[i+1]) for the pairs below.
BANDS = [
    (1e-5, 1e-4),
    (1e-4, 1e-3),
    (0.90, 0.95),
    (0.95, 0.99),
    (0.99, 0.999),
    (0.999, 0.9999),
    (0.9999, 0.99999),
]
REPORT_Q = [
    1e-5,
    1e-4,
    1e-3,
    0.01,
    0.05,
    0.25,
    0.5,
    0.75,
    0.95,
    0.99,
    0.999,
    0.9999,
    0.99999,
]
# Histogram: per sign, bin 0 is |a|/sigma < 1e-3, bins 1..64 are log-spaced up to 1e3, bin 65 is above.
HB = 66
HIST_BINS = 4 * HB  # [open pos, open neg, closed pos, closed neg]
HIST_STRIDE = 8  # main stage: histogram and stats use every 8th position
BAND_STRIDE = 4  # main stage: band sampling uses every 4th position
BAND_EVERY = 2  # main stage: band sampling on every 2nd batch
TOKEN_BUDGET = 32768
CKPT_MIN = 45
LISTS = ["top", "bot", "etop", "ebot"] + [f"band{i}" for i in range(len(BANDS))]
LIST_K = {"top": K_MAIN, "bot": K_MAIN, "etop": K_ECHO, "ebot": K_ECHO} | {
    f"band{i}": K_BAND for i in range(len(BANDS))
}

DEV = torch.device("cuda")
OFF = torch.arange(WIN_LO, WIN_HI + 1, device=DEV)
SHIFT = torch.arange(W, device=DEV, dtype=torch.int64)


def log(*a):
    """Print with a timestamp."""
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------- corpus and batches


def load_corpus(path, model, max_tokens):
    """Return tokens (uint32), offsets (int64) and the number of documents to use."""
    toks = np.load(f"{path}/{model}/tokens.npy", mmap_mode="r")
    offs = np.load(f"{path}/{model}/offsets.npy")
    n = len(offs) - 1
    if max_tokens:
        n = int(np.searchsorted(offs, max_tokens))
        n = max(1, min(n, len(offs) - 1))
    return toks, offs, n


def plan_batches(offs, doc_ids, budget=TOKEN_BUDGET):
    """Group documents (longest first) into batches with B * max_len <= budget."""
    lens = offs[doc_ids + 1] - offs[doc_ids]
    order = doc_ids[np.argsort(-lens, kind="stable")]
    batches, cur, cur_max = [], [], 0
    for d in order:
        L = int(offs[d + 1] - offs[d])
        m = max(cur_max, L)
        if cur and m * (len(cur) + 1) > budget:
            batches.append(np.asarray(cur))
            cur, m = [], L
        cur.append(d)
        cur_max = m
    if cur:
        batches.append(np.asarray(cur))
    return batches


def make_batch(toks, offs, docs):
    """Right-padded input_ids, attention mask, lengths and doc ids on the GPU."""
    lens = (offs[docs + 1] - offs[docs]).astype(np.int64)
    T = int(lens.max())
    ids = np.zeros((len(docs), T), dtype=np.int64)
    for i, d in enumerate(docs):
        ids[i, : lens[i]] = toks[offs[d] : offs[d + 1]]
    ids = torch.from_numpy(ids).to(DEV, non_blocking=True)
    lens_t = torch.from_numpy(lens).to(DEV)
    ar = torch.arange(T, device=DEV)
    attn = (ar[None] < lens_t[:, None]).long()
    valid = (ar[None] < lens_t[:, None]) & (ar[None] >= 1)
    return ids, attn, lens_t, torch.from_numpy(docs.astype(np.int32)).to(DEV), valid


# ---------------------------------------------------------------- model and hooks


def load_model(model):
    """Load the model; return (model, decoder, mlps)."""
    m = AutoModelForCausalLM.from_pretrained(
        MODELS[model], dtype=torch.bfloat16, device_map="cuda"
    )
    m.eval()
    dec = None
    for _, mod in m.named_modules():
        if hasattr(mod, "embed_tokens") and isinstance(
            getattr(mod, "layers", None), nn.ModuleList
        ):
            dec = mod
            break
    assert dec is not None
    mlps = [layer.mlp for layer in dec.layers]
    for mlp in mlps:
        assert (
            hasattr(mlp, "gate_proj")
            and hasattr(mlp, "up_proj")
            and hasattr(mlp, "act_fn")
        )
    return m, dec, mlps


PROC = {"fn": None}


def patch(mlps):
    """Replace each MLP forward with one that sends (g, u, a) to PROC['fn'] first."""
    for li, mlp in enumerate(mlps):

        def fwd(x, mlp=mlp, li=li):
            g = mlp.gate_proj(x)
            u = mlp.up_proj(x)
            a = mlp.act_fn(g) * u
            if PROC["fn"] is not None:
                PROC["fn"](li, g, u, a)
            return mlp.down_proj(a)

        mlp.forward = fwd


# ---------------------------------------------------------------- histogram helpers


def hist_index(a, g, sigma):
    """Histogram bin (int64) for each element. a, g: [M, N]; sigma: [N]."""
    m = (a.float().abs() / sigma).clamp_min(1e-30)
    k = torch.floor((torch.log10(m) + 3.0) * (64 / 6)).add_(1).clamp_(0, HB - 1).long()
    k += (a < 0).long() * HB
    k += (g <= 0).long() * (2 * HB)
    return k


def hist_add(hist, a, g, sigma, chunk=4096):
    """Add [M, N] activations to hist [N, HIST_BINS] (int64)."""
    N = a.shape[1]
    base = torch.arange(N, device=DEV, dtype=torch.int64) * HIST_BINS
    for i in range(0, a.shape[0], chunk):
        k = hist_index(a[i : i + chunk], g[i : i + chunk], sigma) + base
        hist += torch.bincount(k.flatten(), minlength=N * HIST_BINS).view(N, HIST_BINS)


def mag_bounds():
    """Lower and upper |a|/sigma bound of each of the HB bins (bin HB-1 upper bound set later)."""
    e = 10.0 ** (-3.0 + np.arange(65) * 6 / 64)
    lo = np.concatenate([[0.0], e])
    hi = np.concatenate([[1e-3], e[1:], [np.inf]])
    return lo, hi


def hist_quantiles(hist, sigma, vmin, vmax, qs):
    """Quantiles of a from the histogram (gate states combined). Returns [N, len(qs)] float32."""
    h = hist.double()
    c = h[:, : 2 * HB] + h[:, 2 * HB :]
    pos, neg = c[:, :HB], c[:, HB:]
    ordered = torch.cat([neg.flip(1), pos], 1)  # ascending values
    lo, hi = mag_bounds()
    lo_t = torch.tensor(lo, device=hist.device, dtype=torch.float64)
    hi_t = torch.tensor(hi, device=hist.device, dtype=torch.float64)
    s = sigma.double()[:, None]
    top_mag = torch.maximum(vmax.double().abs(), vmin.double().abs())[:, None] / s
    hi_full = hi_t[None].repeat(len(s), 1)
    hi_full[:, -1] = torch.maximum(
        top_mag[:, 0], torch.tensor(1e3, device=hist.device, dtype=torch.float64)
    )
    lo_full = lo_t[None].expand_as(hi_full)
    v_lo = torch.cat([-hi_full.flip(1), lo_full], 1) * s
    v_hi = torch.cat([-lo_full.flip(1), hi_full], 1) * s
    cdf = ordered.cumsum(1)
    tot = cdf[:, -1:].clamp_min(1)
    out = []
    for q in qs:
        tgt = q * tot
        j = torch.searchsorted(cdf, tgt).clamp_(max=2 * HB - 1)
        prev = torch.where(
            j > 0, cdf.gather(1, (j - 1).clamp_min(0)), torch.zeros_like(tgt)
        )
        cnt = ordered.gather(1, j).clamp_min(1)
        f = ((tgt - prev) / cnt).clamp(0, 1)
        out.append(v_lo.gather(1, j) + f * (v_hi.gather(1, j) - v_lo.gather(1, j)))
    return torch.cat(out, 1).float()


# ---------------------------------------------------------------- example lists


def new_list(N, K):
    """Empty example list for N neurons with K slots each."""
    return {
        "score": torch.full((N, K), -math.inf, device=DEV),
        "act": torch.zeros((N, K), device=DEV),
        "doc": torch.full((N, K), -1, device=DEV, dtype=torch.int32),
        "pos": torch.zeros((N, K), device=DEV, dtype=torch.int16),
        "win": torch.full((N, K, W), math.nan, device=DEV, dtype=torch.float16),
        "gbits": torch.zeros((N, K), device=DEV, dtype=torch.int64),
        "pg": torch.zeros((N, K), device=DEV, dtype=torch.float16),
        "pu": torch.zeros((N, K), device=DEV, dtype=torch.float16),
    }


def f16(x):
    """Cast to float16 without overflow to inf."""
    return x.float().clamp(-65000, 65000).half()


def merge(st, cs, cpos, ctx):
    """Merge per-document candidates into list st.

    cs: [B, N] float32 scores (-inf means no candidate); cpos: [B, N] peak positions.
    ctx: dict with A, G, U ([B, T, N]), lens [B], docs [B].
    """
    K = st["score"].shape[1]
    allscore = torch.cat([st["score"], cs.T], 1)
    top, idx = allscore.topk(K, dim=1)
    new = idx >= K
    if not bool(new.any()):
        return
    old = idx.clamp(max=K - 1)
    for key in ("act", "doc", "pos", "gbits", "pg", "pu"):
        st[key] = st[key].gather(1, old)
    st["win"] = st["win"].gather(1, old[..., None].expand(-1, -1, W))
    st["score"] = top
    n_i, k_i = new.nonzero(as_tuple=True)
    b = idx[n_i, k_i] - K
    p = cpos[b, n_i].long()
    A, G, U = ctx["A"], ctx["G"], ctx["U"]
    T = A.shape[1]
    t = p[:, None] + OFF[None]
    inr = (t >= 0) & (t < ctx["lens"][b][:, None])
    tc = t.clamp(0, T - 1)
    bb, nn_ = b[:, None], n_i[:, None]
    w = A[bb, tc, nn_].float().masked_fill(~inr, math.nan)
    gw = G[bb, tc, nn_]
    st["win"][n_i, k_i] = f16(w)
    st["gbits"][n_i, k_i] = (((gw <= 0) & inr).long() << SHIFT[None]).sum(1)
    st["act"][n_i, k_i] = A[b, p, n_i].float()
    st["pg"][n_i, k_i] = f16(G[b, p, n_i])
    st["pu"][n_i, k_i] = f16(U[b, p, n_i])
    st["doc"][n_i, k_i] = ctx["docs"][b]
    st["pos"][n_i, k_i] = p.to(torch.int16)
    empty = torch.isinf(top) & (top < 0)
    st["doc"][empty] = -1


# ---------------------------------------------------------------- stages


def run_batches(dec, toks, offs, batches, label, start=0, on_batch=None):
    """Forward every batch with the current PROC; log progress."""
    t0 = time.time()
    ntok = 0
    total = sum(int((offs[b + 1] - offs[b]).sum()) for b in batches[start:])
    for bi in range(start, len(batches)):
        ids, attn, lens, docs, valid = make_batch(toks, offs, batches[bi])
        CTX.update(lens=lens, docs=docs, valid=valid, bi=bi)
        with torch.no_grad():
            dec(input_ids=ids, attention_mask=attn, use_cache=False)
        ntok += int(lens.sum())
        if on_batch:
            on_batch(bi)
        if bi % 25 == 0 or bi == len(batches) - 1:
            el = time.time() - t0
            rate = ntok / max(el, 1e-6)
            log(
                f"[{label}] batch {bi + 1}/{len(batches)} tok {ntok / 1e6:.2f}M/{total / 1e6:.2f}M "
                f"{rate:.0f} tok/s eta {(total - ntok) / max(rate, 1e-6) / 60:.1f} min "
                f"mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G"
            )


CTX = {}


def stage_sigma(dec, mlps, toks, offs, docs, N):
    """Per-neuron RMS of a over a small document subset."""
    L = len(mlps)
    sumsq = torch.zeros((L, N), device=DEV, dtype=torch.float64)
    cnt = torch.zeros((), device=DEV, dtype=torch.float64)

    def fn(li, g, u, a):
        v = CTX["valid"].reshape(-1)
        x = a.reshape(-1, a.shape[-1])[v].float()
        sumsq[li] += (x * x).sum(0).double()
        if li == 0:
            cnt.add_(x.shape[0])

    PROC["fn"] = fn
    run_batches(dec, toks, offs, plan_batches(offs, docs), "sigma")
    PROC["fn"] = None
    return (sumsq / cnt).sqrt().float().clamp_min(1e-6)


def stage_calib(dec, mlps, toks, offs, docs, sigma):
    """Full-token histograms over a document subset; returns hist, vmin, vmax."""
    L, N = sigma.shape
    hist = torch.zeros((L, N, HIST_BINS), device=DEV, dtype=torch.int64)
    vmin = torch.full((L, N), math.inf, device=DEV)
    vmax = torch.full((L, N), -math.inf, device=DEV)

    def fn(li, g, u, a):
        v = CTX["valid"].reshape(-1)
        x = a.reshape(-1, a.shape[-1])[v]
        gg = g.reshape(-1, g.shape[-1])[v]
        hist_add(hist[li], x, gg, sigma[li])
        vmin[li] = torch.minimum(vmin[li], x.amin(0).float())
        vmax[li] = torch.maximum(vmax[li], x.amax(0).float())

    PROC["fn"] = fn
    run_batches(dec, toks, offs, plan_batches(offs, docs), "calib")
    PROC["fn"] = None
    return hist, vmin, vmax


def new_main_state(L, N):
    """GPU state for the main stage."""
    return {
        "lists": [
            {name: new_list(N, LIST_K[name]) for name in LISTS} for _ in range(L)
        ],
        "hist": torch.zeros((L, N, HIST_BINS), device=DEV, dtype=torch.int64),
        "sum": torch.zeros((L, N), device=DEV, dtype=torch.float64),
        "sumsq": torch.zeros((L, N), device=DEV, dtype=torch.float64),
        "abs_open": torch.zeros((L, N), device=DEV, dtype=torch.float64),
        "abs_closed": torch.zeros((L, N), device=DEV, dtype=torch.float64),
        "n_open": torch.zeros((L, N), device=DEV, dtype=torch.float64),
        "count": torch.zeros((), device=DEV, dtype=torch.float64),
    }


def to_device(obj, dev):
    """Move a nested dict/list of tensors to dev."""
    if isinstance(obj, torch.Tensor):
        return obj.to(dev)
    if isinstance(obj, dict):
        return {k: to_device(v, dev) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_device(v, dev) for v in obj]
    return obj


def stage_main(dec, mlps, toks, offs, docs, sigma, edges, out):
    """Main pass over all documents; returns the GPU state."""
    L, N = sigma.shape
    batches = plan_batches(offs, docs)
    ckpt = f"{out}/main_ckpt.pt"
    start = 0
    if os.path.exists(ckpt):
        saved = torch.load(ckpt, map_location="cpu", weights_only=False)
        S = to_device(saved["state"], DEV)
        start = saved["next_batch"]
        log(f"resumed from checkpoint at batch {start}/{len(batches)}")
        del saved
    else:
        S = new_main_state(L, N)
    lo = edges[:, :, : len(BANDS)]
    hi = edges[:, :, len(BANDS) :]
    ninf, pinf = -math.inf, math.inf

    def fn(li, g, u, a):
        B, T, _ = a.shape
        v3 = CTX["valid"][:, :, None]
        ctx = {"A": a, "G": g, "U": u, "lens": CTX["lens"], "docs": CTX["docs"]}
        lists = S["lists"][li]
        m_open = (g > 0) & v3
        x = a.masked_fill(~m_open, ninf)
        val, pos = x.max(1)
        merge(lists["top"], val.float(), pos, ctx)
        x = a.masked_fill(~m_open, pinf)
        val, pos = x.min(1)
        merge(lists["bot"], -val.float(), pos, ctx)
        m_cl = (~(g > 0)) & v3
        del m_open
        x = a.masked_fill(~m_cl, ninf)
        val, pos = x.max(1)
        merge(lists["etop"], val.float(), pos, ctx)
        x = a.masked_fill(~m_cl, pinf)
        val, pos = x.min(1)
        merge(lists["ebot"], -val.float(), pos, ctx)
        del m_cl, x

        if CTX["bi"] % BAND_EVERY == 0:
            sa = a[:, 1::BAND_STRIDE]
            sv = CTX["valid"][:, 1::BAND_STRIDE, None]
            pr = torch.rand(sa.shape, device=DEV, dtype=torch.float16)
            for bi_ in range(len(BANDS)):
                inb = (
                    (sa >= lo[li, :, bi_].to(sa.dtype))
                    & (sa < hi[li, :, bi_].to(sa.dtype))
                    & sv
                )
                pv, pp = pr.masked_fill(~inb, -1.0).max(1)
                pvf = pv.float()
                pvf = pvf.masked_fill(pvf < 0, ninf)
                merge(lists[f"band{bi_}"], pvf, 1 + pp * BAND_STRIDE, ctx)
            del pr, inb

        vs = CTX["valid"][:, 1::HIST_STRIDE].reshape(-1)
        sa = a[:, 1::HIST_STRIDE].reshape(-1, a.shape[-1])[vs]
        sg = g[:, 1::HIST_STRIDE].reshape(-1, g.shape[-1])[vs]
        hist_add(S["hist"][li], sa, sg, sigma[li])
        xf = sa.float()
        op = sg > 0
        S["sum"][li] += xf.sum(0).double()
        S["sumsq"][li] += (xf * xf).sum(0).double()
        S["n_open"][li] += op.sum(0).double()
        S["abs_open"][li] += (xf.abs() * op).sum(0).double()
        S["abs_closed"][li] += (xf.abs() * ~op).sum(0).double()
        if li == 0:
            S["count"].add_(sa.shape[0])

    last_ckpt = [time.time()]

    def on_batch(bi):
        if time.time() - last_ckpt[0] > CKPT_MIN * 60 and bi < len(batches) - 1:
            t0 = time.time()
            torch.save(
                {"state": to_device(S, "cpu"), "next_batch": bi + 1}, ckpt + ".tmp"
            )
            os.replace(ckpt + ".tmp", ckpt)
            last_ckpt[0] = time.time()
            log(f"checkpoint at batch {bi + 1} ({time.time() - t0:.0f}s)")

    PROC["fn"] = fn
    run_batches(dec, toks, offs, batches, "main", start=start, on_batch=on_batch)
    PROC["fn"] = None
    return S


def stage_logits(model, dec, k=20, chunk=2048):
    """Direct logit effect of each neuron's down_proj column; returns per-layer (ids, vals) for both signs."""
    # Rows past the tokenizer's vocabulary are padding and have no token.
    n_vocab = len(AutoTokenizer.from_pretrained(model.name_or_path))
    WU = model.get_output_embeddings().weight[:n_vocab]
    d = WU.shape[1]
    ones = torch.ones((1, d), device=DEV, dtype=WU.dtype)
    fscale = dec.norm(ones)[0].float()
    res = []
    with torch.no_grad():
        for layer in dec.layers:
            post = getattr(layer, "post_feedforward_layernorm", None)
            pscale = (
                post(ones)[0].float() if post is not None else torch.ones(d, device=DEV)
            )
            Wd = layer.mlp.down_proj.weight
            N = Wd.shape[1]
            ids_p = torch.empty((N, k), dtype=torch.int32, device=DEV)
            val_p = torch.empty((N, k), device=DEV)
            ids_n, val_n = torch.empty_like(ids_p), torch.empty_like(val_p)
            for i in range(0, N, chunk):
                v = (Wd[:, i : i + chunk].float() * (pscale * fscale)[:, None]).to(
                    WU.dtype
                )
                lg = (WU @ v).float()
                tv, ti = lg.topk(k, dim=0)
                bv, bi = (-lg).topk(k, dim=0)
                ids_p[i : i + chunk], val_p[i : i + chunk] = ti.T.int(), tv.T
                ids_n[i : i + chunk], val_n[i : i + chunk] = bi.T.int(), -bv.T
            res.append((ids_p.cpu(), val_p.cpu(), ids_n.cpu(), val_n.cpu()))
    return res


# ---------------------------------------------------------------- output


def save_layer(out, li, S, sigma, edges, qmain, cal_q, logit):
    """Write one layer's raw arrays to raw/layer_XX.npz."""
    arr = {"sigma": sigma[li].cpu().numpy(), "band_edges": edges[li].cpu().numpy()}
    arr["quantiles_main"] = qmain[li].cpu().numpy()
    arr["quantiles_calib"] = cal_q[li].cpu().numpy()
    arr["hist"] = S["hist"][li].cpu().numpy().astype(np.int64)
    for key in ("sum", "sumsq", "abs_open", "abs_closed", "n_open"):
        arr[f"stat_{key}"] = S[key][li].cpu().numpy()
    arr["stat_count"] = np.float64(S["count"].item())
    for name in ("top", "bot", "etop", "ebot"):
        for key, t in S["lists"][li][name].items():
            arr[f"{name}_{key}"] = t.cpu().numpy()
    for key in S["lists"][li]["band0"]:
        arr[f"bands_{key}"] = np.stack(
            [S["lists"][li][f"band{i}"][key].cpu().numpy() for i in range(len(BANDS))],
            1,
        )
    ip, vp, ineg, vneg = logit
    arr.update(
        logit_pos_ids=ip.numpy(),
        logit_pos_vals=vp.numpy(),
        logit_neg_ids=ineg.numpy(),
        logit_neg_vals=vneg.numpy(),
    )
    np.savez(f"{out}/raw/layer_{li:02d}.npz", **arr)


def main():
    """Run all stages for one model."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--corpus", default="/workspace/corpus")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=float, default=0)
    ap.add_argument("--calib-tokens", type=float, default=4e6)
    ap.add_argument("--sigma-tokens", type=float, default=3e5)
    ap.add_argument(
        "--exclude-docs", help="removed_docs.json; these documents are not in the run"
    )
    args = ap.parse_args()
    os.makedirs(f"{args.out}/raw", exist_ok=True)
    t_start = time.time()

    toks, offs, n_docs = load_corpus(args.corpus, args.model, int(args.max_tokens))
    all_docs = np.arange(n_docs)
    removed = None
    if args.exclude_docs:
        removed = json.load(open(args.exclude_docs))
        all_docs = np.setdiff1d(all_docs, np.asarray(removed["docs"], dtype=np.int64))
    all_docs = all_docs[offs[all_docs + 1] > offs[all_docs]]
    n_tok = int((offs[all_docs + 1] - offs[all_docs]).sum())
    log(f"{args.model}: {len(all_docs)} of {n_docs} docs, {n_tok / 1e6:.2f}M tokens")
    model, dec, mlps = load_model(args.model)
    patch(mlps)
    L, N = len(mlps), mlps[0].gate_proj.out_features
    log(f"loaded: {L} layers x {N} neurons, act {type(mlps[0].act_fn).__name__}")

    calib_path = f"{args.out}/calib.pt"
    if os.path.exists(calib_path):
        cal = torch.load(calib_path, map_location=DEV)
        log("calibration loaded")
    else:
        step_c = max(1, round(n_tok / args.calib_tokens))
        step_s = max(1, round(n_tok / args.sigma_tokens))
        sigma = stage_sigma(dec, mlps, toks, offs, all_docs[::step_s], N)
        hist, vmin, vmax = stage_calib(dec, mlps, toks, offs, all_docs[::step_c], sigma)
        qs = sorted({q for b in BANDS for q in b})
        qv = torch.stack(
            [
                hist_quantiles(hist[li], sigma[li], vmin[li], vmax[li], qs)
                for li in range(L)
            ]
        )
        col = {q: i for i, q in enumerate(qs)}
        edges = torch.stack(
            [
                qv[:, :, [col[b[0]] for b in BANDS]],
                qv[:, :, [col[b[1]] for b in BANDS]],
            ],
            2,
        ).reshape(L, N, 2 * len(BANDS))
        cal_q = torch.stack(
            [
                hist_quantiles(hist[li], sigma[li], vmin[li], vmax[li], REPORT_Q)
                for li in range(L)
            ]
        )
        cal = {
            "sigma": sigma,
            "edges": edges,
            "cal_q": cal_q,
            "hist": hist.cpu(),
            "vmin": vmin,
            "vmax": vmax,
        }
        torch.save(cal, calib_path)
        log("calibration done")

    S = stage_main(
        dec, mlps, toks, offs, all_docs, cal["sigma"], cal["edges"], args.out
    )
    t_main = time.time()
    log("main done; logit effects")
    logits = stage_logits(model, dec)
    vmax = torch.stack(
        [
            torch.maximum(
                S["lists"][li]["top"]["act"][:, 0], S["lists"][li]["etop"]["act"][:, 0]
            )
            for li in range(L)
        ]
    )
    vmin = torch.stack(
        [
            torch.minimum(
                S["lists"][li]["bot"]["act"][:, 0], S["lists"][li]["ebot"]["act"][:, 0]
            )
            for li in range(L)
        ]
    )
    qmain = torch.stack(
        [
            hist_quantiles(
                S["hist"][li], cal["sigma"][li], vmin[li], vmax[li], REPORT_Q
            )
            for li in range(L)
        ]
    )
    for li in range(L):
        save_layer(
            args.out, li, S, cal["sigma"], cal["edges"], qmain, cal["cal_q"], logits[li]
        )
    meta = {
        "model": args.model,
        "hf_id": MODELS[args.model],
        "layers": L,
        "neurons_per_layer": N,
        "act_fn": type(mlps[0].act_fn).__name__,
        "n_docs": len(all_docs),
        "n_tokens": n_tok,
        "k": LIST_K,
        "window": [WIN_LO, WIN_HI],
        "bands": BANDS,
        "report_quantiles": REPORT_Q,
        "hist_bins": {
            "per_sign": HB,
            "layout": ["open_pos", "open_neg", "closed_pos", "closed_neg"],
            "log10_range": [-3, 3],
        },
        "hist_stride": HIST_STRIDE,
        "band_stride": BAND_STRIDE,
        "band_every": BAND_EVERY,
        "seconds_total": time.time() - t_start,
        "seconds_main": t_main - t_start,
    }
    if removed is not None:
        meta["removed"] = {
            "docs": len(removed["docs"]),
            "list": "corpus/removed_docs.json",
            "reason": removed["reason"],
            "note": "removed docs are not in the run: no examples, stats or hist",
        }
    json.dump(meta, open(f"{args.out}/meta.json", "w"), indent=1)
    if os.path.exists(f"{args.out}/main_ckpt.pt"):
        os.remove(f"{args.out}/main_ckpt.pt")
    log("DONE")


if __name__ == "__main__":
    main()
