#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
process_news_to_sae_llama3.py
-----------------------------
Build SAE representation vectors for pre-cleaned news CSVs with columns:
- Stock_symbol
- EPSDATS (YYYY-MM-DD)
- Lsa_summary (already-clean text)

Outputs:
- <prefix>_partK_features.npz (X_sum, X_mean, X_max, token_counts, doc_ids)
- <prefix>_partK_meta.csv (lookup by stock_symbol + epsdats)

Example:
python process_news_to_sae_llama3.py \
  --csvs ./data/news/news_2012_agg.csv ./data/news/news_2013_agg.csv \
  --out-root ./data/doc_features/news_llama3_32x \
  --hf-model meta-llama/Meta-Llama-3.1-8B \
  --sae-hub EleutherAI/sae-llama-3.1-8b-32x \
  --hookpoint layers.12 \
  --window 8192 \
  --overlap 128 \
  --batch-flush 100
"""

import os, argparse, logging, re, json
from typing import Optional, List, Tuple

import numpy as np, pandas as pd, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Try EleutherAI SAE library ("sparsify") ---
try:
    from sparsify import Sae  # pip install sparsify
    EAI_SAE_AVAILABLE = True
except Exception:
    EAI_SAE_AVAILABLE = False


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


# ---- Models ----
def load_models(hf_model: str, sae_hub_id: str, hookpoint: str, device: str, hf_token: Optional[str]):
    if not EAI_SAE_AVAILABLE:
        raise RuntimeError("EleutherAI SAE library not available. pip install sparsify")

    # Load Llama 3/3.1
    model = AutoModelForCausalLM.from_pretrained(
        hf_model,
        token=hf_token,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        device_map=None,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(hf_model, token=hf_token)

    # Load the SAE for the desired hookpoint (e.g., "layers.12" or "layers.23.mlp")
    sae = Sae.load_from_hub(sae_hub_id, hookpoint=hookpoint)
    # Move SAE to device for faster encode
    sae = sae.to(device if device.startswith("cuda") else "cpu")

    model.eval(); sae.eval(); torch.set_grad_enabled(False)
    return model, tokenizer, sae


# ---- Hook utilities (only needed for non-residual hookpoints like 'layers.X.mlp') ----
def _resolve_module_by_path(root: torch.nn.Module, path: str) -> torch.nn.Module:
    """
    Resolve dotted module path like 'model.layers.12.mlp' or 'model.model.layers.23.mlp'
    We try a few common prefixes for HF Llama (root is the AutoModelForCausalLM).
    """
    candidates = [
        path,  # if caller already included 'model.model....'
        "model." + path,
        "model.model." + path,
    ]
    for cand in candidates:
        cur = root
        ok = True
        for part in cand.split("."):
            if not hasattr(cur, part):
                ok = False
                break
            cur = getattr(cur, part)
        if ok and isinstance(cur, torch.nn.Module):
            return cur
    raise AttributeError(f"Could not resolve module path for hookpoint='{path}'")


def capture_module_activations(mod: torch.nn.Module, inputs, which: str = "out"):
    """
    Register a forward hook to grab either input ('in') or output ('out') activations.
    Returns activation tensor and removes the hook.
    """
    target_act = {"x": None}

    def hook_fn(m, mod_in, mod_out):
        if which == "in":
            x = mod_in[0] if isinstance(mod_in, (tuple, list)) else mod_in
        else:
            x = mod_out[0] if isinstance(mod_out, (tuple, list)) else mod_out
        target_act["x"] = x
        return mod_out

    h = mod.register_forward_hook(hook_fn)
    _ = inputs  # no-op; we call model(...) outside this helper
    return h, target_act


# ---- Token chunking ----
def chunk_ids(input_ids: torch.Tensor, window: int, overlap: int) -> List[torch.Tensor]:
    T = input_ids.size(1)
    if T <= window:
        return [input_ids]
    chunks = []
    start = 0
    while start < T:
        end = min(T, start + window)
        chunks.append(input_ids[:, start:end])
        if end == T:
            break
        start = max(0, end - overlap)
    return chunks


# ---- Featurization ----
# def _encode_block(sae, activations: torch.Tensor) -> np.ndarray:
#     """
#     activations: [batch, seq, d] or [seq, d] or [n, d] on device
#     returns: np array [T, D_latent]
#     """
#     if activations.dim() == 3:
#         # (B, T, D) -> (B*T, D)
#         activations = activations.flatten(0, 1)
#     elif activations.dim() == 2:
#         pass
#     else:
#         raise ValueError(f"Unexpected activation shape {tuple(activations.shape)}")
#     with torch.no_grad():
#         lat = sae.encode(activations)  # expects (N, D_in); returns (N, D_latent)
#     return lat.detach().cpu().numpy()
def _encode_block(sae, activations: torch.Tensor) -> np.ndarray:
    """
    activations: [batch, seq, d] or [seq, d] on device
    returns: np array [T, D_latent]
    """
    if activations.dim() == 3:
        activations = activations.flatten(0, 1)
    elif activations.dim() != 2:
        raise ValueError(f"Unexpected activation shape {tuple(activations.shape)}")

    with torch.no_grad():
        out = sae.encode(activations)  # newer sparsify returns an EncoderOutput-like object

    # --- unwrap to a tensor (support several field names across versions) ---
    lat = None
    for attr in ("activations", "encoded", "latent", "z", "codes"):
        if hasattr(out, attr):
            lat = getattr(out, attr)
            break
    if lat is None:
        # Sometimes encode may directly return a tensor or a 1-tuple
        if isinstance(out, torch.Tensor):
            lat = out
        elif isinstance(out, (tuple, list)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
            lat = out[0]
        else:
            raise TypeError(f"Unexpected SAE encode() return type: {type(out)}")

    if not isinstance(lat, torch.Tensor):
        raise TypeError(f"Unwrapped SAE output is not a tensor: {type(lat)}")

    return lat.detach().cpu().numpy()


def featurize_text(
    model,
    sae,
    tokenizer,
    text: str,
    device: str,
    hookpoint: str,
    layer_for_residual: Optional[int],
    window: int,
    overlap: int,
    truncate: bool,
    non_residual_capture: bool,
    non_residual_which: str = "out",
):
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=True, truncation=False)
    ids = enc.input_ids.to(device)
    n_tokens = int(ids.size(1))

    if truncate and n_tokens > window:
        ids = ids[:, :window]
        n_tokens = int(ids.size(1))
        chunks = [ids]
    else:
        chunks = chunk_ids(ids, window, overlap)

    sum_vec = None
    max_vec = None
    n_total = 0

    for ch in chunks:
        with torch.inference_mode():
            if not non_residual_capture:
                # Residual-stream path: rely on output_hidden_states to align with hookpoint order
                outputs = model(input_ids=ch, output_hidden_states=True)
                hs = outputs.hidden_states  # tuple: [embed_tokens, layers.0, layers.1, ...]
                if hookpoint == "embed_tokens":
                    acts = torch.as_tensor(hs[0])
                else:
                    # hidden_states[i+1] corresponds to "layers.i"
                    acts = torch.as_tensor(hs[layer_for_residual + 1])
                
                if isinstance(acts, (tuple, list)):
                    acts = acts[0]
            else:
                # Non-residual (e.g., layers.X.mlp or layers.X.attn)
                target_module = _resolve_module_by_path(model, hookpoint)
                hook, bucket = capture_module_activations(target_module, ch, which=non_residual_which)
                _ = model(ch)  # run forward to trigger the hook
                hook.remove()
                acts = bucket["x"]

                if isinstance(acts, (tuple, list)):
                    acts = acts[0]
                if acts is None:
                    raise RuntimeError(f"Failed to capture activations at hookpoint '{hookpoint}'")
            lat_np = _encode_block(sae, acts.to(device).float())  # [T, D_lat]
            if lat_np.size == 0:
                continue

            n_total += lat_np.shape[0]
            if sum_vec is None:
                sum_vec = lat_np.sum(axis=0)
                max_vec = lat_np.max(axis=0)
            else:
                sum_vec += lat_np.sum(axis=0)
                max_vec = np.maximum(max_vec, lat_np.max(axis=0))

            # cleanup
            del acts
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    if n_total == 0:
        # Try to infer D_latent from SAE
        try:
            D = sae.decoder.weight.shape[0]
        except Exception:
            try:
                D = sae.W_dec.weight.shape[0]
            except Exception:
                D = 0
        Z = np.zeros((D,), np.float32)

        return Z, Z.copy(), Z.copy(), 0

    mean_vec = (sum_vec / max(n_total, 1)).astype(np.float32)
    print(sum_vec.shape)
    print(sum_vec)
    return sum_vec.astype(np.float32), mean_vec, max_vec.astype(np.float32), int(n_total)


def normalize_symbol(sym: str) -> str:
    if not isinstance(sym, str):
        return ""
    return re.sub(r"[^A-Za-z0-9._-]", "", sym).upper()


def normalize_date(d: str) -> str:
    # Expecting YYYY-MM-DD; keep as-is if it matches
    if isinstance(d, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", d):
        return d
    # try to coerce (handles e.g. 20120331)
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

    # Model + SAE config
    ap.add_argument("--hf-model", default="meta-llama/Meta-Llama-3.1-8B")
    ap.add_argument("--sae-hub", default="EleutherAI/sae-llama-3.1-8b-32x",
                    help="HF repo id for pretrained Llama3 SAE")
    ap.add_argument("--hookpoint", default="layers.12.mlp",
                    help="Hookpoint string matching the SAE (e.g., 'embed_tokens', 'layers.12', 'layers.23.mlp')")

    # Legacy-like convenience (for residual hookpoints only)
    ap.add_argument("--layer", type=int, default=None,
                    help="(Optional) Layer index for residual stream; if set, overrides hookpoint to 'layers.{layer}'")

    # Windowing
    ap.add_argument("--window", type=int, default=8192, help="Token window for model/SAE")
    ap.add_argument("--overlap", type=int, default=128, help="Token overlap between chunks")
    ap.add_argument("--batch-flush", type=int, default=100, help="Write a part file after N docs")

    # System + columns
    ap.add_argument("--device", default=None, help="e.g., cuda:0 or cpu (default: auto)")
    ap.add_argument("--text-col", default="Lsa_summary")
    ap.add_argument("--symbol-col", default="Stock_symbol")
    ap.add_argument("--date-col", default="EPSDATS")
    ap.add_argument("--truncate", action="store_true", help="If set, truncate to --window instead of chunking")

    # Advanced: when using non-residual modules (mlp/attn), choose whether to grab 'out' or 'in'
    ap.add_argument("--hook-which", choices=["in", "out"], default="out",
                    help="For non-residual hookpoints, use module input or output")

    args = ap.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_root, exist_ok=True)
    logger = make_logger(os.path.join(args.out_root, "process_news.log"))
    logger.info(f"Device: {device}")
    logger.info(f"Mode: {'TRUNCATE' if args.truncate else 'CHUNK+AGG'}")

    # Resolve final hookpoint & whether we can use residual fast-path
    if args.layer is not None:
        hookpoint = f"layers.{args.layer}"
    else:
        hookpoint = args.hookpoint

    residual_fast_path = (hookpoint == "embed_tokens") or bool(re.fullmatch(r"layers\.\d+", hookpoint))
    layer_for_residual = int(hookpoint.split(".")[1]) if residual_fast_path and hookpoint != "embed_tokens" else None

    hf_token = os.getenv("HF_HUB_TOKEN", None)
    model, tokenizer, sae = load_models(args.hf_model, args.sae_hub, hookpoint, device, hf_token)

    # Optional: write an index.json aggregating (symbol, date) -> doc_id across all parts
    global_index = {}

    for csv_path in args.csvs:
        prefix = os.path.splitext(os.path.basename(csv_path))[0]
        logger.info(f"Processing CSV: {csv_path}")

        try:
            df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        except Exception as e:
            logger.exception(f"Failed to read {csv_path}: {e}")
            continue

        # Validate columns
        for col in [args.symbol_col, args.date_col, args.text_col]:
            if col not in df.columns:
                logger.error(f"Missing column '{col}' in {csv_path}; skipping.")
                df = None
                break
        if df is None:
            continue

        X_sum = []; X_mean = []; X_max = []
        token_counts = []; doc_ids = []; info_rows = []
        processed = 0; part = 0

        for idx, row in df.iterrows():
            try:
                sym = normalize_symbol(row.get(args.symbol_col, ""))
                date = normalize_date(row.get(args.date_col, ""))
                text = row.get(args.text_col, "")

                logger.info(f"Processing symbol: {sym}, date: {date}")
                if not text or not sym or not date:
                    logger.warning(f"[{idx}] Skipping empty fields sym={sym} date={date}")
                    continue

                sum_vec, mean_vec, max_vec, ntok = featurize_text(
                    model, sae, tokenizer, text, device,
                    hookpoint=hookpoint,
                    layer_for_residual=layer_for_residual,
                    window=args.window, overlap=args.overlap, truncate=args.truncate,
                    non_residual_capture=not residual_fast_path,
                    non_residual_which=args.hook_which,
                )

                X_sum.append(sum_vec); X_mean.append(mean_vec); X_max.append(max_vec); token_counts.append(ntok)

                # Stable ID & metadata
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
                    "sae_hub": args.sae_hub,
                    "hookpoint": hookpoint,
                    "residual_fast_path": residual_fast_path,
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
                    pd.DataFrame(info_rows).to_csv(
                        os.path.join(args.out_root, f"{prefix}_part{part}_meta.csv"), index=False
                    )
                    logger.info(f"Flushed part {part}: {processed} docs")
                    X_sum.clear(); X_mean.clear(); X_max.clear()
                    token_counts.clear(); doc_ids.clear(); info_rows.clear()
                    if device.startswith("cuda"):
                        torch.cuda.empty_cache()

            except Exception as e:
                logger.exception(f"Row {idx} failed: {e}")
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
                continue

        # Final flush
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
            pd.DataFrame(info_rows).to_csv(
                os.path.join(args.out_root, f"{prefix}_part{part}_meta.csv"), index=False
            )
            logger.info(f"Flushed FINAL part {part}")

        logger.info(f"Finished CSV: {csv_path}")

    # Write a simple global index for quick retrieval by (symbol, epsdats)
    with open(os.path.join(args.out_root, "index_symbol_epsdats.json"), "w", encoding="utf-8") as f:
        json.dump(global_index, f, ensure_ascii=False, indent=2)
    logger.info("Wrote index_symbol_epsdats.json")


if __name__ == "__main__":
    main()
