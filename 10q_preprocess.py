# ingest.py
import asyncio, aiohttp, async_timeout, csv, os, re, html, hashlib, textwrap, time, random, uuid, logging
from bs4 import BeautifulSoup
from pathlib import Path
import re
import boto3
from concurrent.futures import ThreadPoolExecutor


SEC_UA = "sae-research-edgar-pipeline/1.0 (contact: airlay88@gmail.com)"  # per SEC guidance
MAX_CONCURRENCY = 5    # stay well under 10 rps; add jitter/backoff
TIMEOUT = 60

# ITEM_HEADER_PAT = re.compile(
#     r"(?mi)^\s*(?:part\s+[ivxlcdm]+\s*,?\s*)?item\s+(\d{1,2}[aA]?)\s*[\.\-–—:)]"
# )

# --- S3 config (edit these) ---
S3_BUCKET = "njit-sae"
S3_PREFIX_RAW = "10q/raw"               # raw only to S3
S3_PREFIX_CLEAN = "10q/clean_item2"     # cleaned to local + S3

# Thread pool for blocking S3 I/O
_S3_EXEC = ThreadPoolExecutor(max_workers=8)
_S3 = boto3.client("s3")

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
)
log = logging.getLogger("edgar_ingest")

ITEM_HEADER_PAT = re.compile(
    r"(?mi)^\s*(?:part\s+[ivxlcdm]+\s*,?\s*)?item\s+(\d{1,2}[aA]?)\s*[\.\-–—:)]"
)

def extract_primary_item2_from_submission(raw_txt: str):
    blocks = re.findall(r"<DOCUMENT>(.*?)</DOCUMENT>", raw_txt, re.I | re.S)
    primary = None
    for b in blocks:
        m = re.search(r"<TYPE>\s*([^\r\n<]+)", b, re.I)
        if m and m.group(1).strip().upper() in {"10-K","10-Q","8-K","20-F","40-F"}:
            primary = b
            break
    if not primary:
        return None

    m = re.search(r"<TEXT>(.*)", primary, re.I | re.S)
    content = m.group(1) if m else primary

    looks_html = re.search(r"</?(html|table|div|p|span|br)\b", content, re.I)
    if looks_html:
        soup = BeautifulSoup(content, "lxml")
        for t in soup(["script","style","noscript"]): 
            t.decompose()
        text = soup.get_text(separator="\n")
    else:
        text = content

    text = html.unescape(text).replace("\xa0"," ")
    text = re.sub(r"\r", "", text)
    text = re.sub(r"<PAGE>\s*", "", text, flags=re.I)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # find Item 2 sections (ignore TOC by taking longest)
    matches = list(ITEM_HEADER_PAT.finditer(text))
    item2s = []
    for i, m in enumerate(matches):
        label = m.group(1).upper()
        if label != "2": 
            continue
        start, heading_end = m.start(), m.end()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        item2s.append((start, heading_end, end))
    if not item2s:
        return None

    start, heading_end, end = max(item2s, key=lambda t: t[2]-t[0])
    out = (text[start:heading_end] + "\n\n" + text[heading_end:end]).strip()
    out = re.sub(r"\n[ \t]+", "\n", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    return out

async def fetch(session: aiohttp.ClientSession, url: str) -> str | None:
    delay = 1.0
    for attempt in range(6):
        try:
            async with async_timeout.timeout(TIMEOUT):
                log.debug(f"GET {url} (attempt {attempt+1})")
                async with session.get(url) as r:
                    if r.status == 200:
                        text = await r.text()
                        return text
                    if r.status in (403, 429, 503):
                        log.warning(f"{url} -> {r.status}; backing off {delay:.1f}s")
                        await asyncio.sleep(delay + random.random())
                        delay = min(delay * 2, 30)
                    else:
                        log.error(f"{url} -> {r.status}; giving up")
                        return None
        except Exception as e:
            log.warning(f"Fetch error {url}: {e}; backoff {delay:.1f}s")
            await asyncio.sleep(delay + random.random())
            delay = min(delay * 2, 30)
    return None

def _s3_put_bytes_sync(bucket: str, key: str, data: bytes, content_type="text/plain; charset=utf-8"):
    _S3.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)

async def s3_put_text(bucket: str, key: str, text: str):
    data = text.encode("utf-8")
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(_S3_EXEC, _s3_put_bytes_sync, bucket, key, data)

def s3_uri(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key}"

async def worker(rows, out_dir_clean: Path, results, worker_id: int):
    async with aiohttp.ClientSession(headers={"User-Agent": SEC_UA}) as session:
        for idx, row in enumerate(rows, 1):
            url = row["url"]
            year = (row.get("year") or "").strip() or "unknown_year"
            cik  = (row.get("cik")  or "").strip() or "unknown_cik"
            acc  = os.path.basename(row.get("filename","acc.txt")).replace(".txt","")

            row_id = f"[w{worker_id} {idx}/{len(rows)} {year}/{cik}/{acc}]"
            log.info(f"{row_id} start fetch: {url}")
            raw = await fetch(session, url)
            if not raw:
                log.error(f"{row_id} download_failed")
                results.append((row, "download_failed"))
                # gentle pace even on failures
                await asyncio.sleep(0.2 + random.random() * 0.2)
                continue

            # --- RAW -> S3 only ---
            raw_key = f"{S3_PREFIX_RAW}/{year}/{cik}/{acc}.txt"
            try:
                await s3_put_text(S3_BUCKET, raw_key, raw)
                raw_s3 = s3_uri(S3_BUCKET, raw_key)
                log.info(f"{row_id} raw uploaded to {raw_s3}")
            except Exception as e:
                log.error(f"{row_id} raw_s3_upload_failed: {e}")
                results.append((row, "raw_s3_upload_failed"))
                await asyncio.sleep(0.2 + random.random() * 0.2)
                continue

            # --- Extract Item 2 ---
            item2 = extract_primary_item2_from_submission(raw)
            if not item2:
                log.warning(f"{row_id} item2_missing")
                results.append((row, "item2_missing", raw_s3))
                await asyncio.sleep(0.2 + random.random() * 0.2)
                continue

            # --- CLEANED -> local + S3 ---
            # doc_id = str(uuid.uuid4())
            doc_id =  f"{acc}_cleaned"
            clean_rel = f"{year}/{cik}/{doc_id}.txt"
            clean_local = out_dir_clean / clean_rel
            clean_local.parent.mkdir(parents=True, exist_ok=True)

            try:
                clean_local.write_text(item2, encoding="utf-8")
                log.info(f"{row_id} cleaned saved local: {clean_local}")
            except Exception as e:
                log.error(f"{row_id} clean_local_write_failed: {e}")
                results.append((row, "clean_local_write_failed", raw_s3))
                await asyncio.sleep(0.2 + random.random() * 0.2)
                continue

            clean_key = f"{S3_PREFIX_CLEAN}/{clean_rel}"
            try:
                await s3_put_text(S3_BUCKET, clean_key, item2)
                clean_s3 = s3_uri(S3_BUCKET, clean_key)
                log.info(f"{row_id} cleaned uploaded to {clean_s3}")
            except Exception as e:
                log.error(f"{row_id} clean_s3_upload_failed: {e}")
                results.append((row, "clean_s3_upload_failed", raw_s3, str(clean_local)))
                await asyncio.sleep(0.2 + random.random() * 0.2)
                continue

            h = hashlib.sha256(item2.encode("utf-8")).hexdigest()
            results.append((
                row, "ok", doc_id, raw_s3, str(clean_local), clean_s3, h, len(item2)
            ))

            # polite pacing (stay under ~5 rps average)
            await asyncio.sleep(0.2 + random.random() * 0.2)

def chunked(iterable, n):
    buf = []
    for x in iterable:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf

def run_ingest(csv_path, out_root="data", limit=None):
    """
    Returns a list of tuples:
      - on success: (row, "ok", doc_id, raw_s3_uri, clean_local_path, clean_s3_uri, sha256, length)
      - on common failures: (row, status, ...) with partial URIs as available
    """
    out_dir_clean = Path(out_root) / "clean_item2"
    with open(csv_path, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)
    
    if limit is not None:
        rows = rows[:limit]  # or random.sample(rows, limit)

    results = []
    nworkers = min(4, max(1, len(rows)))  # keep your original default while avoiding empty worker sets
    log.info(f"run_ingest: rows={len(rows)} workers={nworkers} max_concurrency={MAX_CONCURRENCY}")

    loop = asyncio.get_event_loop()
    tasks = [
        worker(part, out_dir_clean, results, worker_id=i+1)
        for i, part in enumerate(chunked(rows, max(1, len(rows)//nworkers)))
    ]
    loop.run_until_complete(asyncio.gather(*tasks))
    log.info("run_ingest complete")
    return results


if __name__ == "__main__":
    res = run_ingest("./data/10q/index_2022.csv", out_root="edgar", limit=None)
    print(f"done: {len(res)} rows")