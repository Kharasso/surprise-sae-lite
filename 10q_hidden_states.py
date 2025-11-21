import os
import re
import argparse
import logging
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from sae_text_cleaner import clean_text as external_clean_text
from sae_table_linearizer import linearize_tables

# ----------------- Logging ----------------- #

def make_logger(path):
    logging.basicConfig(
        filename=path,
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("10Q_HiddenStates")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logging.getLogger().addHandler(console)
    return logger

# ----------------- Cleaning utilities ----------------- #

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
    # You already use an external cleaner; this keeps behavior identical
    return external_clean_text(text)
    # If you ever want the fallback:
    # try:
    #     return external_clean_text(text)
    # except Exception:
    #     return fallback_clean_text(text)

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
    # de-duplicate lines, keep order
    seen=set(); merged=[]
    for block in "\n\n".join(lns).splitlines():
        if block not in seen:
            merged.append(block); seen.add(block)
    appendix = "\n\n" + "\n".join(merged) + "\n"
    return cleaned_text + appendix

# ----------------- Path helpers ----------------- #

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def resolve_raw_path(raw_root: str, year: int, cik: str, filename: str) -> str:
    """
    Mirrors your existing 10Q pipeline:
    raw_root / year / cik / <basename>_cleaned.txt
    """
    base = os.path.splitext(os.path.basename(filename))[0] + "_cleaned.txt"
    out_dir = os.path.join(raw_root, str(year), str(cik))
    ensure_dir(out_dir)
    return os.path.join(out_dir, base)

def build_clean_path(clean_root: str, year: int, cik: str, filename: str) -> str:
    """
    Final cleaned text (after your generic cleaner + optional table linearization):
    clean_root / year / cik / <basename>_cleaned.txt
    """
    base = os.path.splitext(os.path.basename(filename))[0] + "_cleaned.txt"
    out_dir = os.path.join(clean_root, str(year), str(cik))
    ensure_dir(out_dir)
    return os.path.join(out_dir, base)

# ----------------- Model + chunking ----------------- #

def chunk_ids(input_ids: torch.Tensor, window: int, overlap: int) -> List[torch.Tensor]:
    """
    Same logic as your SAE pipeline: sliding window with overlap.
    Works for input_ids and attention_mask tensors (shape: [1, T]).
    """
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
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

def featurize_text_hidden_states(
    model,
    tokenizer,
    text: str,
    device: torch.device,
    window: int,
    overlap: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    - Tokenizes the full 10-Q text
    - Chunks into windows (window, overlap)
    - For each chunk:
        * gets last_hidden_state (1, seq_len, hidden_size)
        * takes last token vector as chunk-level "CLS"
        * accumulates token-wise sum for mean pooling
    - Returns:
        cls_doc   : mean of chunk-level CLS vectors (hidden_size,)
        mean_doc  : mean over *all tokens in all chunks* (hidden_size,)
        n_tokens  : total tokens used
    """
    enc = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=False,   # we handle truncation via chunking
    )
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc["attention_mask"].to(device)

    id_chunks = chunk_ids(input_ids, window, overlap)
    mask_chunks = chunk_ids(attn_mask, window, overlap)

    cls_vecs = []
    sum_vec = None
    total_tokens = 0

    for ids_ch, mask_ch in zip(id_chunks, mask_chunks):
        with torch.no_grad():
            outputs = model(input_ids=ids_ch, attention_mask=mask_ch)
        hidden = outputs.last_hidden_state  # (1, seq_len, hidden_size)
        h = hidden[0].detach().cpu().numpy()  # (seq_len, hidden_size)

        # Chunk-level CLS-like vector: last token of the chunk
        cls_vecs.append(h[-1])

        # Accumulate for global mean pooling
        if sum_vec is None:
            sum_vec = h.sum(axis=0)
        else:
            sum_vec += h.sum(axis=0)
        total_tokens += h.shape[0]

        del hidden, outputs
        torch.cuda.empty_cache()

    hidden_size = model.config.hidden_size

    if not cls_vecs or total_tokens == 0:
        # Empty / failed document, return zeros
        return (
            np.zeros((hidden_size,), dtype=np.float32),
            np.zeros((hidden_size,), dtype=np.float32),
            0,
        )

    cls_doc = np.mean(np.stack(cls_vecs, axis=0), axis=0).astype(np.float32)
    mean_doc = (sum_vec / max(total_tokens, 1)).astype(np.float32)

    return cls_doc, mean_doc, total_tokens

# ----------------- Main 10-Q processing ----------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csvs", nargs="+", required=True,
                    help="One or more 10-Q metadata CSVs (same format as your existing pipeline).")
    ap.add_argument("--raw-root", default="./edgar/raw_item2",
                    help="Root dir containing initial _cleaned.txt files (per year/cik).")
    ap.add_argument("--clean-root", default="./edgar/clean_item2",
                    help="Root dir for fully cleaned texts (with optional table linearization).")
    ap.add_argument("--out-root", default="./data/doc_features/10q_9b_hidden_states",
                    help="Output directory for CLS + mean hidden-state features.")
    ap.add_argument("--hf-model", default="google/gemma-2-9b",
                    help="HF model name for Gemma encoder.")
    ap.add_argument("--window", type=int, default=4096,
                    help="Chunk window (tokens) for hidden state extraction.")
    ap.add_argument("--overlap", type=int, default=128,
                    help="Token overlap between consecutive chunks.")
    ap.add_argument("--device", default=None,
                    help="Torch device string (default: auto, cuda:0 if available else cpu).")
    ap.add_argument("--linearize-tables", action="store_true",
                    help="If set, append linearized tables to the cleaned text as in your SAE pipeline.")
    args = ap.parse_args()

    # Device
    device_str = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    torch.set_grad_enabled(False)

    # Dirs & logger
    os.makedirs(args.out_root, exist_ok=True)
    logger = make_logger(os.path.join(args.out_root, "process_10q_hidden_states.log"))
    logger.info(f"Device: {device}")

    # Model + tokenizer (encoder-style, like your transcript script)
    hf_token = os.getenv("HF_HUB_TOKEN", None)
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, trust_remote_code=True, use_auth_token=hf_token)
    model = AutoModel.from_pretrained(args.hf_model, trust_remote_code=True, use_auth_token=hf_token).to(device)
    model.eval()

    for csv_path in args.csvs:
        prefix = os.path.splitext(os.path.basename(csv_path))[0]
        m = re.search(r"(\d{4})", prefix)
        year_guess = int(m.group(1)) if m else None

        logger.info(f"Processing CSV: {csv_path}")

        try:
            meta = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        except Exception as e:
            logger.exception(f"Failed to read {csv_path}: {e}")
            continue

        feats_cls = []
        feats_mean = []
        doc_ids = []
        token_counts = []
        info_rows = []

        for idx, row in meta.iterrows():
            try:
                cik = (row.get("cik") or "").strip()
                year = int(row.get("year") or (year_guess or 0))
                filename = (row.get("filename") or "").strip()
                relative_path = (row.get("relative_path") or "").strip()
                date_filed = (row.get("date_filed") or "").strip()

                # choose something stable as doc id
                base_name = os.path.basename(filename or relative_path or f"row_{idx}")
                doc_id = f"{year}_{cik}_{base_name}"
                logger.info(f"[{prefix}] idx={idx} doc_id={doc_id}")

                # Paths (same logic as your 10-Q SAE pipeline)
                raw_path = resolve_raw_path(args.raw_root, year, cik, filename or relative_path or base_name)
                clean_path = build_clean_path(args.clean_root, year, cik, filename or relative_path or base_name)

                # Prepare cleaned text (on disk) – same semantics you already use
                if not os.path.isfile(clean_path):
                    if not os.path.isfile(raw_path):
                        logger.warning(f"[{idx}] Raw file missing: {raw_path}")
                        continue
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

                # ---- Hidden state featurization ---- #
                cls_vec, mean_vec, n_tok = featurize_text_hidden_states(
                    model=model,
                    tokenizer=tokenizer,
                    text=cleaned,
                    device=device,
                    window=args.window,
                    overlap=args.overlap,
                )

                feats_cls.append(cls_vec)
                feats_mean.append(mean_vec)
                token_counts.append(n_tok)
                doc_ids.append(doc_id)
                info_rows.append(row.to_dict())

            except Exception as e:
                logger.exception(f"[{prefix}] Failed at row {idx}: {e}")
            finally:
                torch.cuda.empty_cache()

        if not feats_cls:
            logger.warning(f"No successful documents for {prefix}, skipping save.")
            continue

        X_cls = np.vstack(feats_cls)
        X_mean = np.vstack(feats_mean)
        doc_ids_arr = np.array(doc_ids, dtype=str)
        token_counts_arr = np.array(token_counts, dtype=np.int32)

        # Save features: CLS-like & mean pooled
        npz_path = os.path.join(args.out_root, f"{prefix}_cls_mean_10q.npz")
        np.savez(
            npz_path,
            X_cls=X_cls,
            X_mean=X_mean,
            doc_ids=doc_ids_arr,
            token_counts=token_counts_arr,
        )
        logger.info(f"Saved feature NPZ: {npz_path}")

        # Save aligned metadata for just the successfully processed docs
        meta_out = pd.DataFrame(info_rows)
        meta_out["doc_id"] = doc_ids
        meta_csv_path = os.path.join(args.out_root, f"{prefix}_cls_mean_10q_meta.csv")
        meta_out.to_csv(meta_csv_path, index=False)
        logger.info(f"Saved metadata CSV: {meta_csv_path}")

if __name__ == "__main__":
    main()
