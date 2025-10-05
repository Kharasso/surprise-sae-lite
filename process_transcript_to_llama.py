import os
import json
import numpy as np
import pandas as pd
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# NEW: SAE-Lens
try:
    from sae_lens import SAE
    SAE_LENS_AVAILABLE = True
except Exception:
    SAE_LENS_AVAILABLE = False

# ——— 1. Single device definition ———
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

logging.basicConfig(
    filename="cls_processing.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
logging.getLogger().addHandler(console)

# === Login & model setup ===
torch.set_grad_enabled(False)
hf_token = os.getenv("HF_HUB_TOKEN")

# ——— 2. Load Llama 3.1 8B + tokenizer ———
model_name = "meta-llama/Llama-3.1-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(
    model_name, token=hf_token, dtype=torch.float32
).to(device)
model.eval()

# ——— 3. Load SAE (Llama-Scope 32×) ———
SCOPE = "res"                # {"res","mlp","att"}
LAYER = 20                   # pick the layer you want
SAE_RELEASE = "llama_scope_lxr_32x" if SCOPE == "res" else \
              ("llama_scope_lxm_32x" if SCOPE == "mlp" else "llama_scope_lxa_32x")
SAE_ID = f"l{LAYER}{ {'res':'r','mlp':'m','att':'a'}[SCOPE] }_32x"

if not SAE_LENS_AVAILABLE:
    raise RuntimeError("sae_lens not available. pip install sae-lens")

sae = SAE.from_pretrained(
    release=SAE_RELEASE,
    sae_id=SAE_ID,
    device="cuda" if device.type == "cuda" else "cpu",
)
sae.eval()

# ——— 4. Hook helper to capture activations at the SAE’s training hookpoint ———
def gather_acts_llama(model, layer: int, inputs, scope: str):
    """
    scope: 'res' -> block output (resid_post)
           'mlp' -> MLP output
           'att' -> attention o_proj output
    Returns [B,T,H] activations as a torch.Tensor.
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

    _ = model(**inputs)  # forward once; hook captures the activations
    for h in handles: h.remove()
    return target  # [B,T,H]

# File lists (UNCHANGED)
jsonl_files = [
    "transcript_componenttext_2012_1.jsonl",
    "transcript_componenttext_2012_2.jsonl",
    "transcript_componenttext_2013_1.jsonl",
    "transcript_componenttext_2013_2.jsonl",
    "transcript_componenttext_2014_1.jsonl",
    "transcript_componenttext_2014_2.jsonl",
]
meta_files = [
    "transcript_metadata_2012_1.csv",
    "transcript_metadata_2012_2.csv",
    "transcript_metadata_2013_1.csv",
    "transcript_metadata_2013_2.csv",
    "transcript_metadata_2014_1.csv",
    "transcript_metadata_2014_2.csv",
]
input_dir = "./data/train_test_data"

jsonl_files = [os.path.join(input_dir, fn) for fn in jsonl_files]
meta_files = [os.path.join(input_dir, fn) for fn in meta_files]

# NEW: output dir name for llama+sae (change as you like)
output_dir = "./data/doc_features/llama31_8b_32x_features"
os.makedirs(output_dir, exist_ok=True)

def process_jsonl(path):
    # UNCHANGED
    temp = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            for key, text in obj.items():
                parts = key.split("_")
                tid, order = parts[2], int(parts[3])
                temp.setdefault(tid, []).append((order, text))
    return {
        tid: "\n".join(str(txt) for _, txt in sorted(lst, key=lambda x: x[0]))
        for tid, lst in temp.items()
    }

# NEW: flush every N docs (like your Gemma script)
FLUSH_EVERY = 100

# Main processing (adds mean/sum/max & periodic flush)
for jpath, mpath in zip(jsonl_files, meta_files):
    prefix = os.path.splitext(os.path.basename(jpath))[0]
    logger.info(f"Starting processing for {prefix}")
    try:
        docs = process_jsonl(jpath)
        tids = list(docs.keys())

        # Load and align metadata (UNCHANGED)
        meta = pd.read_csv(mpath, dtype={"transcriptid": str})
        meta_unique = (
            meta[["transcriptid", "SUESCORE"]]
            .drop_duplicates(subset="transcriptid", keep="first")
            .set_index("transcriptid")
        )

        # --- buffers for mean/sum/max & counts ---
        feats_mean, feats_sum, feats_max = [], [], []
        feats_ntokens = []
        success_tids = []
        count = 0

        for tid in tids:
            logger.info(f"processing tid: {tid}")
            try:
                text = docs[tid]

                # tokenize & truncate to 20k tokens
                tok = tokenizer(
                    text,
                    return_tensors='pt',
                    truncation=True,
                    max_length=20000,
                    add_special_tokens=True
                )
                inputs = {k: v.to(device) for k, v in tok.items()}

                # record token count
                ntok = inputs["input_ids"].size(1)
                feats_ntokens.append(int(ntok))

                with torch.no_grad():
                    # Capture layer activations at the SAE’s hookpoint
                    acts_in = gather_acts_llama(model, LAYER, inputs, scope=SCOPE)  # [1,T,H]
                    # Encode to full latent (dense tensor, mostly zeros)
                    acts_lat = sae.encode(acts_in.to(torch.float32))               # [1,T,D]
                    lat = acts_lat.to(torch.float32).detach().cpu().numpy().squeeze(0)  # [T, D]

                # ---- mean / sum / max over sequence ----
                sum_vec  = lat.sum(axis=0)
                mean_vec = lat.mean(axis=0)
                max_vec  = lat.max(axis=0)

                feats_sum.append(sum_vec)
                feats_mean.append(mean_vec)
                feats_max.append(max_vec)
                success_tids.append(tid)

            except Exception as e:
                logger.exception(f"Failed {tid}: {e}")

            finally:
                if device.type == "cuda":
                    torch.cuda.empty_cache()

                # increment & maybe flush every 100
                count += 1
                if count % FLUSH_EVERY == 0 and success_tids:
                    part = count // FLUSH_EVERY
                    logger.info(f"Flushing part {part} ({count} docs)")
                    npz_path = os.path.join(output_dir, f"{prefix}_part{part}_llama_features.npz")
                    np.savez(
                        npz_path,
                        X_sum=np.vstack(feats_sum),
                        X_mean=np.vstack(feats_mean),
                        X_max=np.vstack(feats_max),
                        token_counts=np.array(feats_ntokens, dtype=int),
                        transcriptids=np.array(success_tids, dtype=str),
                    )
                    # write meta CSV aligned to this batch
                    meta_batch = meta_unique.reindex(success_tids).reset_index()
                    meta_batch["SUESCORE"] = meta_batch["SUESCORE"].astype(float)
                    meta_batch["label"] = meta_batch["SUESCORE"].apply(
                        lambda s: 1 if s >= 0.5 else (0 if s <= -0.5 else np.nan)
                    )
                    meta_csv = os.path.join(output_dir, f"{prefix}_part{part}_llama_features_meta.csv")
                    meta_batch.to_csv(meta_csv, index=False)
                    # clear buffers
                    feats_sum.clear()
                    feats_mean.clear()
                    feats_max.clear()
                    feats_ntokens.clear()
                    success_tids.clear()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

        # final flush for any remainder
        if success_tids:
            part = (count // FLUSH_EVERY) + 1
            logger.info(f"Flushing final part {part} ({len(success_tids)} docs)")
            np.savez(
                os.path.join(output_dir, f"{prefix}_part{part}_llama_features.npz"),
                X_sum=np.vstack(feats_sum),
                X_mean=np.vstack(feats_mean),
                X_max=np.vstack(feats_max),
                token_counts=np.array(feats_ntokens, dtype=int),
                transcriptids=np.array(success_tids, dtype=str),
            )
            meta_batch = meta_unique.reindex(success_tids).reset_index()
            meta_batch["SUESCORE"] = meta_batch["SUESCORE"].astype(float)
            meta_batch["label"] = meta_batch["SUESCORE"].apply(
                lambda s: 1 if s >= 0.5 else (0 if s <= -0.5 else np.nan)
            )
            meta_batch.to_csv(
                os.path.join(output_dir, f"{prefix}_part{part}_llama_features_meta.csv"),
                index=False
            )

        logger.info(f"Finished processing & flushing all parts for {prefix}")

    except Exception as e:
        logger.exception(f"Overall failure for {prefix}: {e}")
