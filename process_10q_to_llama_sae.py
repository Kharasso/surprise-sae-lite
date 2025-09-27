#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
process_10q_to_sae_inmem_llama3_full_latent.py
----------------------------------------------
Build FULL-width SAE feature vectors for 10-Q reports using EleutherAI (sparsify) SAEs
trained on Llama 3/3.1 hookpoints. Cleans and (optionally) linearizes tables in memory;
writes only features (.npz) + metadata (.csv).

Example:
  python process_10q_to_sae_inmem_llama3_full_latent.py \
    --csvs ./data/10q/index_2012.csv ./data/10q/index_2013.csv \
    --raw-root . \
    --out-root ./data/doc_features/10q_llama31_32x \
    --hf-model meta-llama/Meta-Llama-3.1-8B \
    --sae-hub EleutherAI/sae-llama-3.1-8b-32x \
    --layer 20 \
    --window 8192 --overlap 128 --batch-flush 100 \
    --linearize-tables
"""
import os, re, argparse, logging
from typing import Optional, List
import numpy as np, pandas as pd, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Your cleaners (kept as-is)
from sae_text_cleaner import clean_text as external_clean_text
from sae_table_linearizer import linearize_tables

# --- EleutherAI sparsify SAE ---
try:
    from sparsify import Sae  # pip install sparsify
    EAI_SAE_AVAILABLE = True
except Exception:
    EAI_SAE_AVAILABLE = False


# ---------------- Logging ----------------
def make_logger(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    logging.basicConfig(filename=path, filemode="w", level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger("10Q_Pipeline_InMem_Llama31")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logging.getLogger().addHandler(console)
    return logger


# ---------------- Cleaning ----------------
def fallback_clean_text(text: str) -> str:
    import re, unicodedata
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x0c", "")
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    lines = [ln.rstrip().replace("\t", " ") for ln in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    out=[]; L=len(lines); S=text.split("\n")
    for i, ln in enumerate(S):
        s = ln.strip()
        if s.lower() == "table of contents":
            continue
        if s.isdigit() and 1 <= len(s) <= 3:
            prev_blank = (i == 0) or (S[i-1].strip() == "")
            next_blank = (i == L-1) or (S[i+1].strip() == "")
            if prev_blank and next_blank:
                continue
        out.append(ln)
    return "\n".join(out).strip()

def clean_text_generic(text: str) -> str:
    # prefer your external cleaner
    try:
        return external_clean_text(text)
    except Exception:
        return fallback_clean_text(text)

def maybe_append_linearized_tables(cleaned_text: str, enable: bool) -> str:
    if not enable:
        return cleaned_text
    lns = []
    for hint in (None, "Sales:", "Net sales", "Results of Operations"):
        out = linearize_tables(cleaned_text, section_hint=hint) if hint else linearize_tables(cleaned_text)
        if out:
            lns.append(out)
    if not lns:
        return cleaned_text
    seen=set(); merged=[]
    for block in "\n\n".join(lns).splitlines():
        if block not in seen:
            merged.append(block); seen.add(block)
    appendix = "\n\n" + "\n".join(merged) + "\n"
    return cleaned_text + appendix


# ---------------- Paths ----------------
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def resolve_raw_path(raw_root: str, year: int, cik: str, filename: str) -> str:
    base = os.path.splitext(os.path.basename(filename))[0] + "_cleaned.txt"
    out_dir = os.path.join(raw_root, str(year), str(cik)); ensure_dir(out_dir)
    return os.path.join(out_dir, base)

def build_clean_path(clean_root: str, year: int, cik: str, filename: str) -> str:
    base = os.path.splitext(os.path.basename(filename))[0] + "_cleaned.txt"
    out_dir = os.path.join(clean_root, str(year), str(cik)); ensure_dir(out_dir)
    return os.path.join(out_dir, base)


# ---------------- Models ----------------
def load_models(hf_model: str, sae_hub_id: str, hookpoint: str, device: str, hf_token: Optional[str]):
    if not EAI_SAE_AVAILABLE:
        raise RuntimeError("EleutherAI 'sparsify' library not available. Run: pip install sparsify")

    model = AutoModelForCausalLM.from_pretrained(
        hf_model,
        token=hf_token,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        device_map=None,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(hf_model, token=hf_token)

    sae = Sae.load_from_hub(sae_hub_id, hookpoint=hookpoint)
    sae = sae.to(device if device.startswith("cuda") else "cpu")

    model.eval(); sae.eval(); torch.set_grad_enabled(False)
    return model, tokenizer, sae


# ---------------- Hook Utils (for non-residual modules) ----------------
def _resolve_module_by_path(root: torch.nn.Module, path: str) -> torch.nn.Module:
    candidates = [path, "model." + path, "model.model." + path]
    for cand in candidates:
        cur = root; ok = True
        for part in cand.split("."):
            if not hasattr(cur, part):
                ok = False; break
            cur = getattr(cur, part)
        if ok and isinstance(cur, torch.nn.Module):
            return cur
    raise AttributeError(f"Could not resolve module for hookpoint='{path}'")

def capture_module_activations(mod: torch.nn.Module, inputs, which: str = "out"):
    bucket = {"x": None}
    def hook_fn(m, mod_in, mod_out):
        if which == "in":
            x = mod_in[0] if isinstance(mod_in, (tuple, list)) else mod_in
        else:
            x = mod_out[0] if isinstance(mod_out, (tuple, list)) else mod_out
        bucket["x"] = x
        return mod_out
    h = mod.register_forward_hook(hook_fn)
    _ = inputs  # no-op
    return h, bucket


# ---------------- Token chunking ----------------
def chunk_ids(input_ids: torch.Tensor, window: int, overlap: int) -> List[torch.Tensor]:
    T = input_ids.size(1)
    if T <= window:
        return [input_ids]
    chunks = []; start = 0
    while start < T:
        end = min(T, start + window)
        chunks.append(input_ids[:, start:end])
        if end == T:
            break
        start = max(0, end - overlap)
    return chunks


# ---------------- SAE helpers (densify Top-K) ----------------
def _infer_latent_dim_from_sae(sae) -> int:
    for path in ("decoder", "W_dec"):
        mod = getattr(sae, path, None)
        if mod is None: continue
        w = getattr(mod, "weight", None)
        if isinstance(w, torch.Tensor):
            return int(w.shape[0])
    for path in ("cfg", "config"):
        cfg = getattr(sae, path, None)
        if cfg is None: continue
        for k in ("d_sae", "d_out", "n_features", "n_codes"):
            v = getattr(cfg, k, None)
            if isinstance(v, int) and v > 0: return int(v)
    raise RuntimeError("Could not infer SAE latent dimension")

def _encode_block(sae, activations: torch.Tensor) -> np.ndarray:
    """
    activations: [B, T, d] or [T, d]
    returns dense np.ndarray [T, D_latent]
    """
    if activations.dim() == 3:
        activations = activations.flatten(0, 1)
    elif activations.dim() != 2:
        raise ValueError(f"Unexpected activation shape {tuple(activations.shape)}")

    with torch.no_grad():
        out = sae.encode(activations)

    # Dense first (some versions return a tensor or have dense fields)
    for attr in ("activations", "encoded", "latent", "codes"):
        if hasattr(out, attr):
            lat = getattr(out, attr)
            if isinstance(lat, torch.Tensor):
                return lat.detach().cpu().numpy()
    if isinstance(out, torch.Tensor):
        return out.detach().cpu().numpy()
    if isinstance(out, (tuple, list)) and out and isinstance(out[0], torch.Tensor):
        return out[0].detach().cpu().numpy()

    # Sparse Top-K path (indices + values)
    idx = None; vals = None
    for name in ("indices", "idx", "topk_indices", "feature_idx"):
        if hasattr(out, name): idx = getattr(out, name); break
    for name in ("values", "z", "topk_values"):
        if hasattr(out, name): vals = getattr(out, name); break
    if idx is None or vals is None:
        raise TypeError(f"Unexpected SAE encode() output; attrs: {dir(out)}")

    if vals.dim() == 1: vals = vals.unsqueeze(-1)
    if idx.dim() == 1: idx = idx.unsqueeze(-1)

    D_latent = _infer_latent_dim_from_sae(sae)
    N = vals.size(0); device = vals.device
    dense = torch.zeros((N, D_latent), dtype=vals.dtype, device=device)
    dense.scatter_(1, idx.long(), vals)
    return dense.detach().cpu().numpy()


# ---------------- Featurization ----------------
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

    chunks = [ids] if (truncate and n_tokens > window) else chunk_ids(ids, window, overlap)

    sum_vec = None; max_vec = None; n_total = 0

    for ch in chunks:
        with torch.inference_mode():
            if not non_residual_capture:
                # Fast path via hidden_states for residual stream
                outputs = model(input_ids=ch, output_hidden_states=True)
                hs = outputs.hidden_states  # [embed, layer0, layer1, ...]
                if hookpoint == "embed_tokens":
                    acts = torch.as_tensor(hs[0])
                else:
                    acts = torch.as_tensor(hs[layer_for_residual + 1])
                if isinstance(acts, (tuple, list)): acts = acts[0]
            else:
                # Module capture (e.g., layers.X.mlp / .self_attn)
                target = _resolve_module_by_path(model, hookpoint)
                hook, bucket = capture_module_activations(target, ch, which=non_residual_which)
                _ = model(ch)
                hook.remove()
                acts = bucket["x"]
                if isinstance(acts, (tuple, list)): acts = acts[0]
                if acts is None:
                    raise RuntimeError(f"Failed to capture activations at '{hookpoint}'")

            lat_np = _encode_block(sae, acts.to(device).float())  # [T, D_lat]
            if lat_np.size == 0:
                del acts
                if device.startswith("cuda"): torch.cuda.empty_cache()
                continue

            n_total += lat_np.shape[0]
            if sum_vec is None:
                sum_vec = lat_np.sum(axis=0)
                max_vec = lat_np.max(axis=0)
            else:
                sum_vec += lat_np.sum(axis=0)
                max_vec = np.maximum(max_vec, lat_np.max(axis=0))

            del acts, lat_np
            if device.startswith("cuda"): torch.cuda.empty_cache()

    if n_total == 0:
        try:
            D = _infer_latent_dim_from_sae(sae)
        except Exception:
            D = 0
        Z = np.zeros((D,), np.float32)
        return Z, Z.copy(), Z.copy(), 0

    mean_vec = (sum_vec / max(n_total, 1)).astype(np.float32)
    return sum_vec.astype(np.float32), mean_vec, max_vec.astype(np.float32), int(n_total)


# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csvs", nargs="+", required=True)
    ap.add_argument("--raw-root", default="./edgar/raw_item2")
    ap.add_argument("--clean-root", default="./edgar/clean_item2")
    ap.add_argument("--out-root", default="./data/doc_features/10q_llama31_32x")

    # Model + SAE (EleutherAI) config
    ap.add_argument("--hf-model", default="meta-llama/Meta-Llama-3.1-8B")
    ap.add_argument("--sae-hub", default="EleutherAI/sae-llama-3.1-8b-32x",
                    help="HF repo id that contains SAEs per hookpoint")
    ap.add_argument("--hookpoint", default="layers.20",
                    help="Hookpoint matching the SAE, e.g. 'embed_tokens', 'layers.20', 'layers.20.mlp'")

    # Convenience: residual fast path via --layer
    ap.add_argument("--layer", type=int, default=None,
                    help="If set, overrides --hookpoint to 'layers.{layer}' for residual stream")

    # Windowing
    ap.add_argument("--window", type=int, default=8192)
    ap.add_argument("--overlap", type=int, default=128)
    ap.add_argument("--batch-flush", type=int, default=100)

    # System + options
    ap.add_argument("--device", default=None)
    ap.add_argument("--linearize-tables", action="store_true")

    # For non-residual modules, use input vs output capture
    ap.add_argument("--hook-which", choices=["in", "out"], default="out")

    args = ap.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_root, exist_ok=True)
    logger = make_logger(os.path.join(args.out_root, "process_10q_llama31.log"))
    logger.info(f"Device: {device}")

    # Resolve capture point + residual fast path
    hookpoint = f"layers.{args.layer}" if args.layer is not None else args.hookpoint
    residual_fast_path = (hookpoint == "embed_tokens") or bool(re.fullmatch(r"layers\.\d+", hookpoint))
    layer_for_residual = int(hookpoint.split(".")[1]) if (residual_fast_path and hookpoint != "embed_tokens") else None

    hf_token = os.getenv("HF_HUB_TOKEN", None)
    model, tokenizer, sae = load_models(args.hf_model, args.sae_hub, hookpoint, device, hf_token)

    for csv_path in args.csvs:
        prefix = os.path.splitext(os.path.basename(csv_path))[0]
        m = re.search(r"(\d{4})", prefix); year_guess = int(m.group(1)) if m else None
        logger.info(f"Processing CSV: {csv_path}")

        try:
            meta = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        except Exception as e:
            logger.exception(f"Failed to read {csv_path}: {e}"); continue

        X_sum=[]; X_mean=[]; X_max=[]; token_counts=[]; doc_ids=[]; info_rows=[]
        processed=0; part=0

        for idx, row in meta.iterrows():
            try:
                cik = (row.get("cik") or "").strip()
                year = int(row.get("year") or (year_guess or 0))
                filename = (row.get("filename") or "").strip()
                relative_path = (row.get("relative_path") or "").strip()
                date_filed = (row.get("date_filed") or "").strip()

                raw_path = resolve_raw_path(args.raw_root, year, cik, filename or relative_path)
                if raw_path is None:
                    logger.warning(f"[{idx}] Raw not found CIK={cik} file={filename}"); continue

                clean_path = build_clean_path(args.clean_root, year, cik, filename or relative_path)

                if not os.path.isfile(clean_path):
                    with open(raw_path, "r", encoding="utf-8", errors="ignore") as f:
                        raw_text = f.read()
                    cleaned = clean_text_generic(raw_text)
                    cleaned = maybe_append_linearized_tables(cleaned, args.linearize_tables)
                    with open(clean_path, "w", encoding="utf-8") as f:
                        f.write(cleaned)
                else:
                    with open(clean_path, "r", encoding="utf-8", errors="ignore") as f:
                        cleaned = f.read()
                    if args.linearize_tables and "TABLE_LINEARIZED" not in cleaned:
                        cleaned2 = maybe_append_linearized_tables(cleaned, True)
                        if cleaned2 != cleaned:
                            with open(clean_path, "w", encoding="utf-8") as f:
                                f.write(cleaned2)
                            cleaned = cleaned2

                # ---- Encode to FULL latent width ----
                sum_vec, mean_vec, max_vec, ntok = featurize_text(
                    model, sae, tokenizer, cleaned, device,
                    hookpoint=hookpoint,
                    layer_for_residual=layer_for_residual,
                    window=args.window, overlap=args.overlap, truncate=False,  # 10-Qs: usually chunked
                    non_residual_capture=not residual_fast_path,
                    non_residual_which=args.hook_which,
                )

                X_sum.append(sum_vec); X_mean.append(mean_vec); X_max.append(max_vec); token_counts.append(ntok)

                # Identity + metadata
                base_name = os.path.splitext(os.path.basename(filename or relative_path or raw_path))[0]
                doc_id = f"{cik}_{base_name}"
                doc_ids.append(doc_id)
                info_rows.append({
                    "doc_id": doc_id, "cik": cik, "company": row.get("company",""), "form": row.get("form",""),
                    "date_filed": date_filed, "quarter": row.get("quarter",""), "year": year,
                    "url": row.get("url",""), "raw_path": raw_path, "clean_path": clean_path,
                    "ntokens": ntok, "used_linearizer": bool(args.linearize_tables),
                    "hf_model": args.hf_model, "sae_hub": args.sae_hub,
                    "hookpoint": hookpoint, "residual_fast_path": residual_fast_path,
                    "window": args.window, "overlap": args.overlap
                })

                processed += 1
                if processed % args.batch_flush == 0:
                    part += 1
                    npz_path = os.path.join(args.out_root, f"{prefix}_part{part}_features.npz")
                    np.savez(
                        npz_path,
                        X_sum=np.vstack(X_sum), X_mean=np.vstack(X_mean), X_max=np.vstack(X_max),
                        token_counts=np.array(token_counts, np.int32), doc_ids=np.array(doc_ids, dtype=object),
                    )
                    pd.DataFrame(info_rows).to_csv(
                        os.path.join(args.out_root, f"{prefix}_part{part}_meta.csv"), index=False
                    )
                    logging.getLogger("10Q_Pipeline_InMem_Llama31").info(f"Flushed part {part}: {processed} docs")
                    X_sum.clear(); X_mean.clear(); X_max.clear(); token_counts.clear(); doc_ids.clear(); info_rows.clear()
                    if device.startswith("cuda"): torch.cuda.empty_cache()

            except Exception as e:
                logging.getLogger("10Q_Pipeline_InMem_Llama31").exception(f"Row {idx} failed: {e}")
                if device.startswith("cuda"): torch.cuda.empty_cache()
                continue

        if len(doc_ids) > 0:
            part += 1
            npz_path = os.path.join(args.out_root, f"{prefix}_part{part}_features.npz")
            np.savez(
                npz_path,
                X_sum=np.vstack(X_sum), X_mean=np.vstack(X_mean), X_max=np.vstack(X_max),
                token_counts=np.array(token_counts, np.int32), doc_ids=np.array(doc_ids, dtype=object),
            )
            pd.DataFrame(info_rows).to_csv(
                os.path.join(args.out_root, f"{prefix}_part{part}_meta.csv"), index=False
            )
            logging.getLogger("10Q_Pipeline_InMem_Llama31").info(f"Flushed FINAL part {part}")
        logging.getLogger("10Q_Pipeline_InMem_Llama31").info(f"Finished CSV: {csv_path}")


if __name__ == "__main__":
    main()
