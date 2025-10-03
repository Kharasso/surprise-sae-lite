#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
process_news_to_sae_llama31.py
------------------------------
Build full-width (131k) SAE latent vectors for pre-cleaned news CSVs using
Llama-3.1-8B 32x SAEs (Llama-Scope).

Defaults target residual stream SAEs:
  release = "llama_scope_lxr_32x"
  sae_id  = f"l{layer}r_32x"  (auto-derived from --layer)

Columns expected in CSVs:
  - Stock_symbol
  - EPSDATS           (YYYY-MM-DD)
  - Lsa_summary       (already-clean text)

Outputs:
  - <prefix>_partK_features.npz  (X_sum, X_mean, X_max, token_counts, doc_ids)
  - <prefix>_partK_meta.csv      (lookup by stock_symbol + epsdats)

Example:
  python process_news_to_sae_llama31.py \
    --csvs ./data/news/news_2012_agg.csv ./data/news/news_2013_agg.csv \
    --out-root ./data/doc_features/news_llama31_32x_res \
    --hf-model meta-llama/Llama-3.1-8B \
    --sae-release llama_scope_lxr_32x \
    --layer 12 \
    --window 8192 \
    --overlap 128 \
    --batch-flush 100
"""

import os, argparse, logging, re, json
from typing import Optional, List
import numpy as np, pandas as pd, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from sae_lens import SAE
    SAE_LENS_AVAILABLE = True
except Exception:
    SAE_LENS_AVAILABLE = False


# ---- Logging ----
def make_logger(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    logging.basicConfig(
        filename=path,
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("NEWS_SAE_Pipeline")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logging.getLogger().addHandler(console)
    return logger


# ---- Model & SAE ----
def build_sae_id(scope: str, layer: int) -> str:
    """Return the SAELens sae_id for Llama-Scope 32x families.
    scope in {"res","mlp","att"} maps to r/m/a."""
    tag = {"res": "r", "mlp": "m", "att": "a"}[scope]
    return f"l{layer}{tag}_32x"

def load_models(hf_model: str, sae_release: str, sae_id: str, device: str, hf_token: Optional[str]):
    if not SAE_LENS_AVAILABLE:
        raise RuntimeError("sae_lens not available. `pip install sae-lens`.")
    model = AutoModelForCausalLM.from_pretrained(hf_model, use_auth_token=hf_token).to(device)
    tokenizer = AutoTokenizer.from_pretrained(hf_model, use_auth_token=hf_token)
    sae, _, _ = SAE.from_pretrained(
        release=sae_release,   # e.g., "llama_scope_lxr_32x"
        sae_id=sae_id,         # e.g., "l12r_32x"
        device=device if device.startswith("cuda") else "cpu",
    )
    model.eval(); sae.eval(); torch.set_grad_enabled(False)
    return model, tokenizer, sae


# ---- Hooks (residual, mlp_out, attn_out) ----
def gather_acts_llama(model, layer: int, inputs, scope: str):
    """
    scope: 'res'  -> block output  (resid_post)
           'mlp'  -> MLP output    (hook_mlp_out)
           'att'  -> attn output   (hook_attn_out / o_proj out)
    Returns [B, T, H] activations matching SAE training hookpoint.
    """
    target = None

    def hook_block(_, __, out):
        nonlocal target
        tgt = out[0] if isinstance(out, (tuple, list)) else out
        target = tgt
        return out

    def hook_mlp(_, __, out):
        nonlocal target
        tgt = out[0] if isinstance(out, (tuple, list)) else out
        target = tgt
        return out

    def hook_attn_o(_, __, out):
        nonlocal target
        tgt = out[0] if isinstance(out, (tuple, list)) else out
        target = tgt
        return out

    handles = []
    layer_mod = model.model.layers[layer]

    if scope == "res":
        handles.append(layer_mod.register_forward_hook(hook_block))
    elif scope == "mlp":
        handles.append(layer_mod.mlp.register_forward_hook(hook_mlp))
    elif scope == "att":
        # Llama uses self_attn.o_proj for the output projection
        handles.append(layer_mod.self_attn.o_proj.register_forward_hook(hook_attn_o))
    else:
        raise ValueError("scope must be one of {'res','mlp','att'}")

    _ = model(inputs)
    for h in handles: h.remove()
    return target


# ---- Token chunking ----
def chunk_ids(input_ids: torch.Tensor, window: int, overlap: int) -> List[torch.Tensor]:
    T = input_ids.size(1)
    if T <= window:
        return [input_ids]
    chunks = []; start = 0
    while start < T:
        end = min(T, start + window)
        chunks.append(input_ids[:, start:end])
        if end == T: break
        start = max(0, end - overlap)
    return chunks


# ---- Featurization (dense 131k; no top-k trimming) ----
def featurize_text(model, sae, tokenizer, text: str, device: str,
                   layer: int, window: int, overlap: int, truncate: bool, scope: str):
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=True, truncation=False)
    ids = enc.input_ids.to(device)
    n_tokens = int(ids.size(1))

    if truncate and n_tokens > window:
        ids = ids[:, :window]
        n_tokens = int(ids.size(1))
        chunks = [ids]
    else:
        chunks = chunk_ids(ids, window, overlap)

    sum_vec = None; max_vec = None; n_total = 0
    for ch in chunks:
        with torch.no_grad():
            acts_in = gather_acts_llama(model, layer, ch, scope=scope)   # [B,T,H]
            # SAE.encode returns full-width dense activations shaped [B,T,D_sae] (sparse but dense tensor)
            acts_lat = sae.encode(acts_in.float())
            arr = acts_lat.detach().cpu().numpy().squeeze(0)  # [T, D_sae]
        if arr.size == 0:
            continue
        n_total += arr.shape[0]
        block_sum = arr.sum(axis=0)
        block_max = arr.max(axis=0)
        if sum_vec is None:
            sum_vec = block_sum; max_vec = block_max
        else:
            sum_vec += block_sum; max_vec = np.maximum(max_vec, block_max)
        del acts_in, acts_lat
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    if n_total == 0:
        D = int(getattr(sae, "d_sae", 0) or (sae.W_dec.weight.shape[0] if hasattr(sae, "W_dec") else 0))
        Z = np.zeros((D,), np.float32)
        return Z, Z.copy(), Z.copy(), 0

    mean_vec = (sum_vec / max(n_total, 1)).astype(np.float32)
    return sum_vec.astype(np.float32), mean_vec, max_vec.astype(np.float32), int(n_total)


def normalize_symbol(sym: str) -> str:
    if not isinstance(sym, str):
        return ""
    return re.sub(r"[^A-Za-z0-9._-]", "", sym).upper()

def normalize_date(d: str) -> str:
    if isinstance(d, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", d):
        return d
    try:
        d = str(d)
        if re.fullmatch(r"\d{8}", d):
            return f"{d[:4]}-{d[4:6]}-{d[6:]}"
    except Exception:
        pass
    return str(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csvs", nargs="+", required=True, help="Paths to news_{year}_agg.csv files")
    ap.add_argument("--out-root", required=True, help="Directory to write features/meta")

    # Llama-3.1 8B + Llama-Scope 32x defaults
    ap.add_argument("--hf-model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--sae-release", default="llama_scope_lxr_32x",
                    help="SAELens release name (e.g., llama_scope_lxr_32x / lxm / lxa)")
    ap.add_argument("--sae-id", default=None,
                    help="If omitted, auto-derived from --layer and --scope (e.g., l12r_32x)")
    ap.add_argument("--scope", choices=["res", "mlp", "att"], default="res",
                    help="Hookpoint family: residual (res), mlp (mlp), or attention output (att)")

    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--window", type=int, default=8192, help="Token window for model/SAE")
    ap.add_argument("--overlap", type=int, default=128, help="Token overlap between chunks")
    ap.add_argument("--batch-flush", type=int, default=100, help="Write a part file after N docs")
    ap.add_argument("--device", default=None, help="e.g., cuda:0 or cpu (default: auto)")
    ap.add_argument("--text-col", default="Lsa_summary")
    ap.add_argument("--symbol-col", default="Stock_symbol")
    ap.add_argument("--date-col", default="EPSDATS")
    ap.add_argument("--truncate", action="store_true", help="If set, truncate to --window instead of chunking")
    ap.add_argument("--max-rows", type=int, default=None,
                help="If set, only process the first N rows of each CSV (per file).")
    args = ap.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_root, exist_ok=True)
    logger = make_logger(os.path.join(args.out_root, "process_news_llama31.log"))
    logger.info(f"Device: {device}")
    logger.info(f"Mode: {'TRUNCATE' if args.truncate else 'CHUNK+AGG'}")
    logger.info(f"Scope: {args.scope}")

    hf_token = os.getenv("HF_HUB_TOKEN", None)
    sae_id = args.sae_id or build_sae_id(args.scope, args.layer)
    model, tokenizer, sae = load_models(args.hf_model, args.sae_release, sae_id, device, hf_token)
    logger.info(f"Loaded SAE release='{args.sae_release}', sae_id='{sae_id}' (d_sae≈{getattr(sae, 'd_sae', 'unknown')})")

    global_index = {}

    for csv_path in args.csvs:
        prefix = os.path.splitext(os.path.basename(csv_path))[0]
        logger.info(f"Processing CSV: {csv_path}")

        try:
            df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)

            if args.max_rows is not None:
                df = df.head(args.max_rows)
                logger.info(f"Limiting to first {args.max_rows} rows for {csv_path}")
        except Exception as e:
            logger.exception(f"Failed to read {csv_path}: {e}")
            continue

        for col in [args.symbol_col, args.date_col, args.text_col]:
            if col not in df.columns:
                logger.error(f"Missing column '{col}' in {csv_path}; skipping.")
                df = None
                break
        if df is None:
            continue

        X_sum = []; X_mean = []; X_max = []; token_counts = []; doc_ids = []; info_rows = []
        processed = 0; part = 0

        for idx, row in df.iterrows():
            try:
                sym = normalize_symbol(row.get(args.symbol_col, ""))
                date = normalize_date(row.get(args.date_col, ""))
                text = row.get(args.text_col, "")
                if not text or not sym or not date:
                    logging.warning(f"[{idx}] Skipping empty fields sym={sym} date={date}")
                    continue

                sum_vec, mean_vec, max_vec, ntok = featurize_text(
                    model, sae, tokenizer, text, device, args.layer, args.window, args.overlap, args.truncate, args.scope
                )
                X_sum.append(sum_vec); X_mean.append(mean_vec); X_max.append(max_vec); token_counts.append(ntok)

                doc_id = f"{sym}_{date}"
                doc_ids.append(doc_id)
                info_rows.append({
                    "doc_id": doc_id,
                    "stock_symbol": sym,
                    "epsdats": date,
                    "ntokens": ntok,
                    "mode": "truncate" if args.truncate else "chunk",
                    "window": args.window,
                    "overlap": args.overlap if not args.truncate else 0,
                    "hf_model": args.hf_model,
                    "sae_release": args.sae_release,
                    "sae_id": sae_id,
                    "scope": args.scope,
                    "layer": args.layer,
                    "source_csv": csv_path,
                })
                global_index[f"{sym}||{date}"] = doc_id
                processed += 1

                if processed % args.batch_flush == 0:
                    part += 1
                    npz_path = os.path.join(args.out_root, f"{prefix}_part{part}_features.npz")
                    np.savez(
                        npz_path,
                        X_sum=np.vstack(X_sum),
                        X_mean=np.vstack(X_mean),
                        X_max=np.vstack(X_max),
                        token_counts=np.array(token_counts, np.int32),
                        doc_ids=np.array(doc_ids, dtype=object),
                    )
                    pd.DataFrame(info_rows).to_csv(os.path.join(args.out_root, f"{prefix}_part{part}_meta.csv"), index=False)
                    logger.info(f"Flushed part {part}: {processed} docs")
                    X_sum.clear(); X_mean.clear(); X_max.clear(); token_counts.clear(); doc_ids.clear(); info_rows.clear()
                    if device.startswith("cuda"):
                        torch.cuda.empty_cache()

            except Exception as e:
                logger.exception(f"Row {idx} failed: {e}")
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
                continue

        if len(doc_ids) > 0:
            part += 1
            npz_path = os.path.join(args.out_root, f"{prefix}_part{part}_features.npz")
            np.savez(
                npz_path,
                X_sum=np.vstack(X_sum),
                X_mean=np.vstack(X_mean),
                X_max=np.vstack(X_max),
                token_counts=np.array(token_counts, np.int32),
                doc_ids=np.array(doc_ids, dtype=object),
            )
            pd.DataFrame(info_rows).to_csv(os.path.join(args.out_root, f"{prefix}_part{part}_meta.csv"), index=False)
            logger.info(f"Flushed FINAL part {part}")
        logger.info(f"Finished CSV: {csv_path}")

    with open(os.path.join(args.out_root, "index_symbol_epsdats.json"), "w", encoding="utf-8") as f:
        json.dump(global_index, f, ensure_ascii=False, indent=2)
    logger.info("Wrote index_symbol_epsdats.json")


if __name__ == "__main__":
    main()
