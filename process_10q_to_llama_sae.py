#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
process_10q_to_sae_llama31.py
-----------------------------
Same behavior as your original Gemma pipeline re: raw/clean I/O:
- Open the raw TXT from --raw-root (or direct path fields from CSV).
- If the cleaned file is NOT in --clean-root, clean + (optionally) linearize, then WRITE it there.
- Always read/featurize the cleaned text.

Only changes:
- Switch to Llama-3.1-8B with Llama-Scope 32x SAEs (full 131,072-d latent).
- SAE scope/layer handling (res/mlp/att), auto-build sae_id.
- bfloat16 -> float32 safety when converting to NumPy.
- Optional --tail-tokens and --max-rows (per-CSV).
- Optional --torch-dtype for HF model load.

Example:
  python process_10q_to_sae_llama31.py \
    --csvs ./data/10q/index_2012_sp500.csv ./data/10q/index_2013_sp500.csv ./data/10q/index_2014_sp500.csv \
    --raw-root ./edgar/raw_item2 \
    --clean-root ./edgar/clean_item2 \
    --out-root ./data/doc_features/10q_llama31_8b_32x \
    --hf-model meta-llama/Llama-3.1-8B \
    --sae-release llama_scope_lxr_32x \
    --scope res \
    --layer 20 \
    --window 8192 \
    --overlap 128 \
    --batch-flush 100 \
    --linearize-tables \
    --tail-tokens 20000 \
    --max-rows 50
"""
import os, re, argparse, logging
from typing import Optional, List
import numpy as np, pandas as pd, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_text_cleaner import clean_text as external_clean_text
from sae_table_linearizer import linearize_tables

try:
    from sae_lens import SAE
    SAE_LENS_AVAILABLE = True
except Exception:
    SAE_LENS_AVAILABLE = False


# ---------- Logging ----------
def make_logger(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    logging.basicConfig(
        filename=path, filemode="w", level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger("10Q_Pipeline_LLAMA31")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logging.getLogger().addHandler(console)
    return logger


# ---------- Cleaning ----------
def fallback_clean_text(text: str) -> str:
    import unicodedata
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


# ---------- Paths (PRESERVED BEHAVIOR) ----------
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def resolve_raw_path(raw_root: str, year: int, cik: str, filename: str) -> str:
    base = os.path.splitext(os.path.basename(filename))[0] + "_cleaned.txt"
    out_dir = os.path.join(raw_root, str(year), str(cik)); ensure_dir(out_dir)
    return os.path.join(out_dir, base)

def build_clean_path(clean_root: str, year: int, cik: str, filename: str) -> str:
    base = os.path.splitext(os.path.basename(filename))[0] + "_cleaned.txt"
    out_dir = os.path.join(clean_root, str(year), str(cik)); ensure_dir(out_dir)
    return os.path.join(out_dir, base)

# def ensure_dir(p): os.makedirs(p, exist_ok=True)

# def resolve_raw_path(row: pd.Series, root: str) -> Optional[str]:
#     """
#     Original behavior: look for raw file in --raw-root using any of:
#     ['local_path', 'relative_path', 'filename'] (if present).
#     Try <root>/<value> and the bare <value> as fallback.
#     """
#     cands=[]
#     for col in ["local_path", "relative_path", "filename"]:
#         if col in row and pd.notna(row[col]):
#             p=str(row[col]).replace("\\","/").strip()
#             if p:
#                 cands.append(os.path.join(root, p))
#                 cands.append(p)
#     for c in cands:
#         if os.path.isfile(c):
#             return c
#     return None

# def build_clean_path(clean_root: str, row: pd.Series) -> str:
#     """
#     Mirror your prior convention: /clean_root/<year>/<cik>/<basename>_cleaned.txt
#     where basename comes from 'filename' or 'relative_path' or 'local_path'.
#     """
#     year = str(row.get("year") or "")
#     cik  = str(row.get("cik") or "")
#     # pick a name source in priority order
#     name = (row.get("filename") or row.get("relative_path") or row.get("local_path") or "doc.txt")
#     base = os.path.splitext(os.path.basename(str(name)))[0] + "_cleaned.txt"
#     out_dir = os.path.join(clean_root, year, cik); ensure_dir(out_dir)
#     return os.path.join(out_dir, base)


# ---------- Llama-Scope SAE ----------
def build_sae_id(scope: str, layer: int) -> str:
    tag = {"res": "r", "mlp": "m", "att": "a"}[scope]
    return f"l{layer}{tag}_32x"

def str2dtype(s):
    return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}.get(s, None)

def load_models(hf_model: str, sae_release: str, sae_id: str, device: str,
                hf_token: Optional[str], torch_dtype_opt: str = "auto"):
    if not SAE_LENS_AVAILABLE:
        raise RuntimeError("sae_lens not available. `pip install sae-lens`.")
    torch_dtype = str2dtype(torch_dtype_opt)  # None => auto
    model = AutoModelForCausalLM.from_pretrained(
        hf_model, use_auth_token=hf_token, torch_dtype=torch_dtype
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(hf_model, use_auth_token=hf_token)
    sae = SAE.from_pretrained(
        release=sae_release, sae_id=sae_id,
        device=device if device.startswith("cuda") else "cpu",
    )
    model.eval(); sae.eval(); torch.set_grad_enabled(False)
    return model, tokenizer, sae


# ---------- Hooking ----------
def gather_acts_llama(model, layer: int, inputs, scope: str):
    """
    scope: 'res' -> block output (resid_post)
           'mlp' -> MLP output
           'att' -> attention o_proj output
    Returns [B,T,H] activations.
    """
    target = None
    def _cap(_, __, out):
        nonlocal target
        target = out[0] if isinstance(out, (tuple, list)) else out
        return out

    handles = []
    layer_mod = model.model.layers[layer]
    if scope == "res":
        handles.append(layer_mod.register_forward_hook(_cap))
    elif scope == "mlp":
        handles.append(layer_mod.mlp.register_forward_hook(_cap))
    elif scope == "att":
        handles.append(layer_mod.self_attn.o_proj.register_forward_hook(_cap))
    else:
        raise ValueError("scope must be one of {'res','mlp','att'}")

    _ = model(inputs)
    for h in handles: h.remove()
    return target


# ---------- Chunking & Featurization ----------
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

def featurize_text(model, sae, tokenizer, text: str, device: str,
                   layer: int, window: int, overlap: int, scope: str,
                   tail_tokens: Optional[int] = None):
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=True, truncation=False)
    ids = enc.input_ids.to(device)
    if tail_tokens is not None and ids.size(1) > tail_tokens:
        ids = ids[:, -tail_tokens:]

    chunks = chunk_ids(ids, window, overlap)

    sum_vec = None; max_vec = None; n_total = 0
    for ch in chunks:
        with torch.no_grad():
            acts_in = gather_acts_llama(model, layer, ch, scope=scope)        # [B,T,H]
            acts_lat = sae.encode(acts_in.to(torch.float32))                  # [B,T,131072]
            arr = acts_lat.to(torch.float32).detach().cpu().numpy().squeeze(0)
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


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csvs", nargs="+", required=True)
    ap.add_argument("--raw-root", default="./edgar/raw_item2")
    ap.add_argument("--clean-root", default="./edgar/clean_item2")
    ap.add_argument("--out-root", default="./data/doc_features/10q_llama31_8b_32x")
    ap.add_argument("--hf-model", default="meta-llama/Llama-3.1-8B")

    # Llama-Scope 32x defaults (residual)
    ap.add_argument("--sae-release", default="llama_scope_lxr_32x",
                    help="Choose among llama_scope_lxr_32x (res), llama_scope_lxm_32x (mlp), llama_scope_lxa_32x (att)")
    ap.add_argument("--sae-id", default=None,
                    help="If omitted, auto from --layer and --scope (e.g., l20r_32x)")
    ap.add_argument("--scope", choices=["res", "mlp", "att"], default="res",
                    help="Hookpoint family (residual/mlp/attn)")

    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--window", type=int, default=8192)
    ap.add_argument("--overlap", type=int, default=128)
    ap.add_argument("--batch-flush", type=int, default=100)
    ap.add_argument("--device", default=None)
    ap.add_argument("--linearize-tables", action="store_true")

    # Extras to match your news pipeline
    ap.add_argument("--max-rows", type=int, default=None,
                    help="If set, only process the first N rows of each CSV (per file).")
    ap.add_argument("--tail-tokens", type=int, default=None,
                    help="If set, keep only the last N tokens per document before chunking.")
    ap.add_argument("--torch-dtype", choices=["auto","float32","bfloat16","float16"], default="auto",
                    help="Torch dtype to load the HF model with")

    args = ap.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_root, exist_ok=True)
    logger = make_logger(os.path.join(args.out_root, "process_10q_llama31.log"))
    logger.info(f"Device: {device}")
    logger.info(f"Scope: {args.scope}")

    hf_token = os.getenv("HF_HUB_TOKEN", None)
    sae_id = args.sae_id or build_sae_id(args.scope, args.layer)
    model, tokenizer, sae = load_models(
        args.hf_model, args.sae_release, sae_id, device, hf_token, args.torch_dtype
    )
    logger.info(f"Loaded SAE release='{args.sae_release}', sae_id='{sae_id}' (d_sae≈{getattr(sae,'d_sae','unknown')})")

    for csv_path in args.csvs:
        prefix = os.path.splitext(os.path.basename(csv_path))[0]
        m = re.search(r"(\d{4})", prefix); year_guess = int(m.group(1)) if m else None
        logger.info(f"Processing CSV: {csv_path}")

        try:
            meta = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
            if args.max_rows is not None:
                meta = meta.head(args.max_rows)
                logger.info(f"Limiting to first {args.max_rows} rows for {csv_path}")
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

                # ---- ORIGINAL I/O BEHAVIOR ----
                name_for_paths = (row.get("filename") or row.get("relative_path") or row.get("local_path") or "doc.txt")

                raw_path = resolve_raw_path(args.raw_root, year, cik, name_for_paths)
                if raw_path is None:
                    logger.warning(f"[{idx}] Raw not found CIK={cik} file={filename or relative_path}"); continue

                clean_path = build_clean_path(args.clean_root, year, cik, name_for_paths)

                # If not already cleaned in clean-root, produce and write it there
                if not os.path.isfile(clean_path):
                    with open(raw_path, "r", encoding="utf-8", errors="ignore") as f:
                        raw_text = f.read()
                    cleaned = clean_text_generic(raw_text)
                    cleaned = maybe_append_linearized_tables(cleaned, args.linearize_tables)
                    with open(clean_path, "w", encoding="utf-8") as f:
                        f.write(cleaned)
                # Always read from clean-root for featurization
                with open(clean_path, "r", encoding="utf-8", errors="ignore") as f:
                    cleaned = f.read()

                # ---- Featurize (full 131k; no top-k) ----
                sum_vec, mean_vec, max_vec, ntok = featurize_text(
                    model, sae, tokenizer, cleaned, device,
                    args.layer, args.window, args.overlap, args.scope,
                    tail_tokens=args.tail_tokens
                )
                X_sum.append(sum_vec); X_mean.append(mean_vec); X_max.append(max_vec); token_counts.append(ntok)

                # Identity
                chosen_name = filename or relative_path or row.get("local_path") or os.path.basename(raw_path)
                doc_id = f"{cik}_{os.path.splitext(os.path.basename(str(chosen_name)))[0]}"
                doc_ids.append(doc_id)
                info_rows.append({
                    "doc_id": doc_id,
                    "cik": cik,
                    "company": row.get("company",""),
                    "form": row.get("form",""),
                    "date_filed": date_filed,
                    "quarter": row.get("quarter",""),
                    "year": year,
                    "url": row.get("url",""),
                    "raw_path": raw_path,
                    "clean_path": clean_path,
                    "ntokens": ntok,
                    "used_linearizer": bool(args.linearize_tables),
                    "hf_model": args.hf_model,
                    "sae_release": args.sae_release,
                    "sae_id": sae_id,
                    "scope": args.scope,
                    "layer": args.layer
                })

                processed += 1
                logger.info(f"Processed file: {raw_path}")

                if processed % args.batch_flush == 0:
                    part += 1
                    npz_path = os.path.join(args.out_root, f"{prefix}_part{part}_features.npz")
                    np.savez(npz_path,
                             X_sum=np.vstack(X_sum), X_mean=np.vstack(X_mean), X_max=np.vstack(X_max),
                             token_counts=np.array(token_counts, np.int32), doc_ids=np.array(doc_ids, dtype=object))
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
            np.savez(npz_path,
                     X_sum=np.vstack(X_sum), X_mean=np.vstack(X_mean), X_max=np.vstack(X_max),
                     token_counts=np.array(token_counts, np.int32), doc_ids=np.array(doc_ids, dtype=object))
            pd.DataFrame(info_rows).to_csv(os.path.join(args.out_root, f"{prefix}_part{part}_meta.csv"), index=False)
            logger.info(f"Flushed FINAL part {part}")
        logger.info(f"Finished CSV: {csv_path}")


if __name__ == "__main__":
    main()
