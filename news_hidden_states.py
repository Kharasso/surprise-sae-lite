#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
process_news_to_hidden_states.py
--------------------------------
Build Gemma-2 hidden-state vectors for pre-cleaned news CSVs with columns:
  - Stock_symbol
  - EPSDATS           (YYYY-MM-DD)
  - Lsa_summary       (already-clean text)

Outputs (per CSV, possibly multiple parts):
  - <prefix>_partK_hidden.npz  (X_cls, X_mean, token_counts, doc_ids)
  - <prefix>_partK_meta.csv    (lookup by stock_symbol + epsdats)
  - index_symbol_epsdats.json  (global mapping (symbol,date) -> doc_id)

Example:
  python process_news_to_hidden_states.py \
    --csvs ./data/news/news_2012_agg.csv ./data/news/news_2013_agg.csv \
    --out-root ./data/doc_features/news_gemma2_9b_hidden \
    --hf-model google/gemma-2-9b \
    --window 8192 \
    --overlap 128 \
    --batch-flush 100
"""

import os
import argparse
import logging
import re
import json
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


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
    logger = logging.getLogger("NEWS_HIDDEN_STATES")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logging.getLogger().addHandler(console)
    return logger


# ---- Token chunking (same logic as SAE script) ----
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


# ---- Hidden-state featurization ----
def featurize_text_hidden_states(
    model,
    tokenizer,
    text: str,
    device: str,
    window: int,
    overlap: int,
    truncate: bool,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Returns:
      - cls_doc   : document-level CLS-like vector (mean of chunk-level last-token vectors)
      - mean_doc  : mean-pooled hidden state over all tokens in all chunks
      - n_tokens  : total tokens used
    """
    enc = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=False,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    n_tokens = int(input_ids.size(1))

    if truncate and n_tokens > window:
        input_ids = input_ids[:, :window]
        attention_mask = attention_mask[:, :window]
        n_tokens = int(input_ids.size(1))
        id_chunks = [input_ids]
        mask_chunks = [attention_mask]
    else:
        id_chunks = chunk_ids(input_ids, window, overlap)
        mask_chunks = chunk_ids(attention_mask, window, overlap)

    cls_vecs = []
    sum_vec = None
    total_tokens = 0

    for ids_ch, mask_ch in zip(id_chunks, mask_chunks):
        with torch.no_grad():
            outputs = model(input_ids=ids_ch, attention_mask=mask_ch)
        hidden = outputs.last_hidden_state  # (1, seq_len, hidden_size)
        h = hidden[0].detach().cpu().numpy()  # (seq_len, hidden_size)

        # CLS-like: use last token of the chunk
        cls_vecs.append(h[-1])

        # Accumulate for global mean
        if sum_vec is None:
            sum_vec = h.sum(axis=0)
        else:
            sum_vec += h.sum(axis=0)
        total_tokens += h.shape[0]

        del hidden, outputs
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    hidden_size = model.config.hidden_size

    if not cls_vecs or total_tokens == 0:
        # Fallback zeros
        Z = np.zeros((hidden_size,), np.float32)
        return Z, Z.copy(), 0

    cls_doc = np.mean(np.stack(cls_vecs, axis=0), axis=0).astype(np.float32)
    mean_doc = (sum_vec / max(total_tokens, 1)).astype(np.float32)
    return cls_doc, mean_doc, total_tokens


# ---- Normalization helpers (copied from your SAE script) ----
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
    ap.add_argument("--hf-model", default="google/gemma-2-9b")
    ap.add_argument("--window", type=int, default=8192, help="Token window for model")
    ap.add_argument("--overlap", type=int, default=128, help="Token overlap between chunks")
    ap.add_argument("--batch-flush", type=int, default=100, help="Write a part file after N docs")
    ap.add_argument("--device", default=None, help="e.g., cuda:0 or cpu (default: auto)")
    ap.add_argument("--text-col", default="Lsa_summary")
    ap.add_argument("--symbol-col", default="Stock_symbol")
    ap.add_argument("--date-col", default="EPSDATS")
    ap.add_argument("--truncate", action="store_true",
                    help="If set, truncate to --window instead of chunking")
    args = ap.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_root, exist_ok=True)
    logger = make_logger(os.path.join(args.out_root, "process_news_hidden_states.log"))
    logger.info(f"Device: {device}")
    logger.info(f"Mode: {'TRUNCATE' if args.truncate else 'CHUNK+AGG'}")

    hf_token = os.getenv("HF_HUB_TOKEN", None)
    tokenizer = AutoTokenizer.from_pretrained(
        args.hf_model,
        trust_remote_code=True,
        use_auth_token=hf_token,
    )
    model = AutoModel.from_pretrained(
        args.hf_model,
        trust_remote_code=True,
        use_auth_token=hf_token,
    ).to(device)
    model.eval()
    torch.set_grad_enabled(False)

    # Optional: index (symbol, date) -> doc_id across all parts
    global_index = {}

    for csv_path in args.csvs:
        prefix = os.path.splitext(os.path.basename(csv_path))[0]  # e.g., news_2012_agg
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

        X_cls = []
        X_mean = []
        token_counts = []
        doc_ids = []
        info_rows = []

        processed = 0
        part = 0

        for idx, row in df.iterrows():
            try:
                sym = normalize_symbol(row.get(args.symbol_col, ""))
                date = normalize_date(row.get(args.date_col, ""))
                text = row.get(args.text_col, "")
                logger.info(f"Processing symbol: {sym}, date: {date}")

                if not text or not sym or not date:
                    logger.warning(f"[{idx}] Skipping empty fields sym={sym} date={date}")
                    continue

                cls_vec, mean_vec, ntok = featurize_text_hidden_states(
                    model=model,
                    tokenizer=tokenizer,
                    text=text,
                    device=device,
                    window=args.window,
                    overlap=args.overlap,
                    truncate=args.truncate,
                )

                X_cls.append(cls_vec)
                X_mean.append(mean_vec)
                token_counts.append(ntok)

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
                    "source_csv": csv_path,
                })

                global_index[f"{sym}||{date}"] = doc_id
                processed += 1

                # Batch flush (same pattern as SAE script)
                if processed % args.batch_flush == 0:
                    part += 1
                    npz_path = os.path.join(args.out_root, f"{prefix}_part{part}_hidden.npz")
                    np.savez(
                        npz_path,
                        X_cls=np.vstack(X_cls),
                        X_mean=np.vstack(X_mean),
                        token_counts=np.array(token_counts, np.int32),
                        doc_ids=np.array(doc_ids, dtype=object),
                    )
                    meta_path = os.path.join(args.out_root, f"{prefix}_part{part}_meta.csv")
                    pd.DataFrame(info_rows).to_csv(meta_path, index=False)
                    logger.info(f"Flushed part {part}: {processed} docs")

                    X_cls.clear()
                    X_mean.clear()
                    token_counts.clear()
                    doc_ids.clear()
                    info_rows.clear()
                    if str(device).startswith("cuda"):
                        torch.cuda.empty_cache()

            except Exception as e:
                logger.exception(f"Row {idx} failed: {e}")
                if str(device).startswith("cuda"):
                    torch.cuda.empty_cache()
                continue

        # Final flush
        if len(doc_ids) > 0:
            part += 1
            npz_path = os.path.join(args.out_root, f"{prefix}_part{part}_hidden.npz")
            np.savez(
                npz_path,
                X_cls=np.vstack(X_cls),
                X_mean=np.vstack(X_mean),
                token_counts=np.array(token_counts, np.int32),
                doc_ids=np.array(doc_ids, dtype=object),
            )
            meta_path = os.path.join(args.out_root, f"{prefix}_part{part}_meta.csv")
            pd.DataFrame(info_rows).to_csv(meta_path, index=False)
            logger.info(f"Flushed FINAL part {part}")
        logger.info(f"Finished CSV: {csv_path}")

    # Global index
    index_path = os.path.join(args.out_root, "index_symbol_epsdats.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(global_index, f, ensure_ascii=False, indent=2)
    logger.info(f"Wrote {index_path}")


if __name__ == "__main__":
    main()
