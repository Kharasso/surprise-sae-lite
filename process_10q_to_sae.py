#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
process_10q_to_sae_inmem.py
---------------------------
Same as process_10q_to_sae.py, but performs *all* text cleaning and optional
table linearization in memory. It does NOT write any cleaned text files.
Only feature .npz files and metadata CSVs are written.

Run example:
  python process_10q_to_sae_inmem.py \
    --csvs ./data/10q/index_2012.csv ./data/10q/index_2013.csv ./data/10q/index_2014.csv \
    --raw-root . \
    --out-root ./data/doc_features/10q_9b_canonical_131k \
    --hf-model google/gemma-2-9b \
    --sae-release gemma-scope-9b-pt-res-canonical \
    --sae-id layer_20/width_131k/canonical \
    --layer 20 --window 4096 --overlap 128 --batch-flush 100 \
    --linearize-tables
"""
import os, re, argparse, logging
from typing import Optional
import numpy as np, pandas as pd, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_text_cleaner import clean_text as external_clean_text
from sae_table_linearizer import linearize_tables

try:
    from sae_lens import SAE
    SAE_LENS_AVAILABLE = True
except Exception:
    SAE_LENS_AVAILABLE = False

def make_logger(path):
    logging.basicConfig(filename=path, filemode="w", level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger("10Q_Pipeline_InMem")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logging.getLogger().addHandler(console)
    return logger

# ---- Cleaner hookup ----
# def try_import_cleaner():
#     try:
#         from sae_text_cleaner import clean_text as external_clean_text
#         return external_clean_text
#     except Exception:
#         return None

def fallback_clean_text(text: str) -> str:
    import re, unicodedata
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x0c", "")
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    lines = [ln.rstrip().replace("\t", " ") for ln in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # conservative artifacts
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
    # ext = try_import_cleaner()
    # if ext is not None:
    #     return ext(text)
    # return fallback_clean_text(text)
    return external_clean_text(text)

# ---- Table linearization (optional) ----
# def try_import_linearizer():
#     try:
#         from sae_table_linearizer import linearize_tables
#         return linearize_tables
#     except Exception:
#         return None

def maybe_append_linearized_tables(cleaned_text: str, enable: bool) -> str:
    if not enable:
        return cleaned_text
    # fn = try_import_linearizer()
    # if fn is None:
    #     return cleaned_text
    lns = []
    for hint in (None, "Sales:", "Net sales", "Results of Operations"):
        out = linearize_tables(cleaned_text, section_hint=hint) if hint else linearize_tables(cleaned_text)
        if out:
            lns.append(out)
    if not lns:
        return cleaned_text
    # de-duplicate lines, keep order
    seen=set(); merged=[]
    for block in "\n\n".join(lns).splitlines():
        if block not in seen:
            merged.append(block); seen.add(block)
    appendix = "\n\n" + "\n".join(merged) + "\n"
    return cleaned_text + appendix

# ---- Paths ----
def ensure_dir(p): os.makedirs(p, exist_ok=True)

# def resolve_raw_path(row: pd.Series, root: str) -> Optional[str]:
#     cands=[]
#     for col in ["local_path","relative_path","filename"]:
#         if col in row and pd.notna(row[col]):
#             p=str(row[col]).replace("\\","/")
#             cands.append(os.path.join(root,p)); cands.append(p)
#     for c in cands:
#         if os.path.isfile(c): return c
#     return None
def resolve_raw_path(raw_root: str, year: int, cik: str, filename: str) -> str:
    base = os.path.splitext(os.path.basename(filename))[0] + "_cleaned.txt"
    out_dir = os.path.join(raw_root, str(year), str(cik)); ensure_dir(out_dir)
    return os.path.join(out_dir, base)

def build_clean_path(clean_root: str, year: int, cik: str, filename: str) -> str:
    base = os.path.splitext(os.path.basename(filename))[0] + "_cleaned.txt"
    out_dir = os.path.join(clean_root, str(year), str(cik)); ensure_dir(out_dir)
    return os.path.join(out_dir, base)

# ---- Models ----
def load_models(hf_model: str, sae_release: str, sae_id: str, device: str, hf_token: Optional[str]):
    model = AutoModelForCausalLM.from_pretrained(hf_model, use_auth_token=hf_token).to(device)
    tokenizer = AutoTokenizer.from_pretrained(hf_model, use_auth_token=hf_token)
    if not SAE_LENS_AVAILABLE: raise RuntimeError("sae_lens not available. `pip install sae-lens`.")
    sae, _, _ = SAE.from_pretrained(release=sae_release, sae_id=sae_id, device=device if device.startswith("cuda") else "cpu")
    model.eval(); sae.eval(); torch.set_grad_enabled(False)
    return model, tokenizer, sae

# ---- Hook ----
def gather_residual_activations(model, target_layer, inputs):
    target_act=None
    def hook(mod, mod_in, mod_out):
        nonlocal target_act
        target_act = mod_out[0] if isinstance(mod_out,(tuple,list)) else mod_out
        return mod_out
    h = model.model.layers[target_layer].register_forward_hook(hook)
    _ = model(inputs)
    h.remove()
    return target_act

# ---- Chunking/featurize ----
def chunk_ids(input_ids: torch.Tensor, window: int, overlap: int):
    T=input_ids.size(1)
    if T<=window: return [input_ids]
    chunks=[]; start=0
    while start<T:
        end=min(T, start+window)
        chunks.append(input_ids[:, start:end])
        if end==T: break
        start=end-overlap
        if start<0: start=0
    return chunks

def featurize_text(model, sae, tokenizer, text: str, device: str, layer: int, window: int, overlap: int):
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=True).input_ids.to(device)
    chunks = chunk_ids(ids, window, overlap)
    sum_vec=None; max_vec=None; n_total=0
    for ch in chunks:
        with torch.no_grad():
            res = gather_residual_activations(model, layer, ch)
            acts = sae.encode(res.float())
            arr = acts.detach().cpu().numpy().squeeze(0)
        if arr.size==0: continue
        n_total += arr.shape[0]
        if sum_vec is None:
            sum_vec = arr.sum(axis=0); max_vec = arr.max(axis=0)
        else:
            sum_vec += arr.sum(axis=0); max_vec = np.maximum(max_vec, arr.max(axis=0))
        del res, acts; torch.cuda.empty_cache()
    if n_total==0:
        D = sae.W_dec.weight.shape[0] if hasattr(sae,"W_dec") else 0
        return np.zeros((D,), np.float32), np.zeros((D,), np.float32), np.zeros((D,), np.float32), 0
    mean_vec = (sum_vec / max(n_total,1)).astype(np.float32)
    return sum_vec.astype(np.float32), mean_vec, max_vec.astype(np.float32), int(n_total)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--csvs", nargs="+", required=True)
    ap.add_argument("--raw-root", default="./edgar/raw_item2")
    ap.add_argument("--clean-root", default="./edgar/clean_item2")
    ap.add_argument("--out-root", default="./data/doc_features/10q_9b_canonical_131k")
    ap.add_argument("--hf-model", default="google/gemma-2-9b")
    ap.add_argument("--sae-release", default="gemma-scope-9b-pt-res-canonical")
    ap.add_argument("--sae-id", default="layer_20/width_131k/canonical")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--window", type=int, default=4096)
    ap.add_argument("--overlap", type=int, default=128)
    ap.add_argument("--batch-flush", type=int, default=100)
    ap.add_argument("--device", default=None)
    ap.add_argument("--linearize-tables", action="store_true")
    args=ap.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_root, exist_ok=True)
    logger = make_logger(os.path.join(args.out_root, "process_10q.log"))
    logger.info(f"Device: {device}")
    hf_token = os.getenv("HF_HUB_TOKEN", None)
    model, tokenizer, sae = load_models(args.hf_model, args.sae_release, args.sae_id, device, hf_token)

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

                # raw_path = resolve_raw_path(row, args.raw_root)
                # if raw_path is None:
                #     logger.warning(f"[{idx}] Raw not found CIK={cik} file={filename}"); continue

                # raw_path = build_clean_path(args.clean_root, year, cik, filename or relative_path or raw_path)
                # if raw_path is None:
                #     logger.warning(f"[{idx}] Raw not found CIK={cik} file={filename}"); continue

                # # In-memory clean + optional linearization
                # with open(raw_path, "r", encoding="utf-8", errors="ignore") as f:
                #     raw_text = f.read()
                # cleaned = clean_text_generic(raw_text)
                # if args.linearize_tables:
                #     cleaned = maybe_append_linearized_tables(cleaned, True)

                raw_path = resolve_raw_path(args.raw_root, year, cik, filename or relative_path or raw_path)
                if raw_path is None:
                    logger.warning(f"[{idx}] Raw not found CIK={cik} file={filename}"); continue

                clean_path = build_clean_path(args.clean_root, year, cik, filename or relative_path or raw_path)

                if not os.path.isfile(clean_path):
                    with open(raw_path, "r", encoding="utf-8", errors="ignore") as f: raw_text=f.read()
                    cleaned = clean_text_generic(raw_text)
                    cleaned = maybe_append_linearized_tables(cleaned, args.linearize_tables)
                    with open(clean_path, "w", encoding="utf-8") as f: f.write(cleaned)
                else:
                    with open(clean_path, "r", encoding="utf-8", errors="ignore") as f: cleaned=f.read()
                    if args.linearize_tables and "TABLE_LINEARIZED" not in cleaned:
                        # Append linearizations if they weren't added previously
                        cleaned2 = maybe_append_linearized_tables(cleaned, True)
                        if cleaned2 != cleaned:
                            with open(clean_path, "w", encoding="utf-8") as f: f.write(cleaned2)
                            cleaned = cleaned2

                # Featurize
                sum_vec, mean_vec, max_vec, ntok = featurize_text(
                    model, sae, tokenizer, cleaned, device, args.layer, args.window, args.overlap
                )
                X_sum.append(sum_vec); X_mean.append(mean_vec); X_max.append(max_vec); token_counts.append(ntok)

                # Identity
                doc_id = f"{cik}_{os.path.splitext(os.path.basename(filename or relative_path or raw_path))[0]}"
                doc_ids.append(doc_id)
                info_rows.append({
                    "doc_id": doc_id, "cik": cik, "company": row.get("company",""), "form": row.get("form",""),
                    "date_filed": date_filed, "quarter": row.get("quarter",""), "year": year,
                    "url": row.get("url",""), "raw_path": raw_path, "ntokens": ntok,
                    "used_linearizer": bool(args.linearize_tables)
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
                    torch.cuda.empty_cache()

            except Exception as e:
                logger.exception(f"Row {idx} failed: {e}")
                torch.cuda.empty_cache(); continue

        if len(doc_ids)>0:
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
