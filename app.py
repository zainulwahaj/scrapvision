from flask import Flask, request, jsonify
from flask_cors import CORS
import asyncio
import trafilatura
from summa import summarizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from cachetools import LRUCache, cached
import httpx
import uuid
import logging
import psutil
import time
from threading import Thread
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup

# -----------------------------
#     App Initialization
# -----------------------------
app = Flask(__name__)

# Allow CORS on every route with a wildcard origin
CORS(app, resources={r"/*": {"origins": "*"}})

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# -----------------------------
#     Configuration
# -----------------------------
MIN_TEXT_LENGTH = 200
DEFAULT_LIMIT = 10         # Default # of results if none provided
CONCURRENT_WORKERS = 10    # Number of concurrent worker tasks
cache = LRUCache(maxsize=5000)

# -----------------------------
#  Sentiment Analyzer (VADER)
# -----------------------------
sentiment_analyzer = SentimentIntensityAnalyzer()

# -----------------------------
#  In‐Memory Job Storage
# -----------------------------
jobs = {}
jobs_lock = asyncio.Lock()

async def create_job():
    job_id = str(uuid.uuid4())
    async with jobs_lock:
        jobs[job_id] = {"status": "in_progress", "results": [], "error": None}
    return job_id

async def update_job(job_id, result=None, status=None, error=None):
    async with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return
        if result is not None:
            job["results"].append(result)
        if status:
            job["status"] = status
        if error:
            job["error"] = error

async def get_job(job_id):
    async with jobs_lock:
        return jobs.get(job_id)

# -----------------------------
#  Text Processing (cached)
# -----------------------------
@cached(cache)
def analyze_content(text: str):
    """
    1) Summarize with summa.TextRank (fallback to first 3 sentences).
    2) Sentiment via VADER → compound in [−1,1] → map to confidence ∈ [0.0,1.0].
    3) Map that confidence to star buckets (1..5).
    """
    # ---- Summarization (TextRank via summa) ----
    try:
        summary = summarizer.summarize(text, ratio=0.1)
        if not summary.strip():
            raise ValueError
    except:
        # Fallback: first 3 sentences
        summary = ". ".join(text.split(". ")[:3]) + "."

    # ---- Sentiment via VADER ----
    scores = sentiment_analyzer.polarity_scores(text)
    comp = scores["compound"]  # in [−1.0 … +1.0]

    # ---- Rescale compound to confidence ∈ [0.0 … 1.0] ----
    conf_float = (comp + 1.0) / 2.0  # 0.0 when comp=−1.0, 1.0 when comp=+1.0

    # ---- Map confidence to 1..5 star buckets ----
    if conf_float >= 0.80:
        star_label = "5 stars"
    elif conf_float >= 0.60:
        star_label = "4 stars"
    elif conf_float >= 0.40:
        star_label = "3 stars"
    elif conf_float >= 0.20:
        star_label = "2 stars"
    else:
        star_label = "1 star"

    # ---- Sentiment label ----
    if comp > 0:
        sentiment_label = "POSITIVE"
    elif comp < 0:
        sentiment_label = "NEGATIVE"
    else:
        sentiment_label = "NEUTRAL"

    return summary, {
        "star_label": star_label,
        "sentiment_label": sentiment_label,
        # Return confidence as a float 0.0..1.0 (two decimal places)
        "sentiment_score": round(conf_float, 3)
    }

# -----------------------------
#  Memory Usage Logger (optional)
# -----------------------------
def log_memory_usage():
    proc = psutil.Process()
    while True:
        mb = proc.memory_info().rss / (1024**2)
        logger.warning(f"Memory usage: {mb:.1f} MB")
        time.sleep(30)

Thread(target=log_memory_usage, daemon=True).start()

# -----------------------------
#  URL Normalization / Validation
# -----------------------------
def normalize_url(url: str) -> str:
    # Strip fragment (#...), lowercase scheme & host, strip trailing slash
    url = url.split("#")[0]
    p = urlparse(url)
    norm = p._replace(scheme=p.scheme.lower(), netloc=p.netloc.lower())
    return norm.geturl().rstrip("/")

def is_valid_url(url: str, domain: str = None) -> bool:
    try:
        p = urlparse(url)
        if p.scheme not in ("http", "https") or not p.netloc:
            return False
        if domain and p.netloc != domain:
            return False
        return True
    except:
        return False

# -----------------------------
#  Single‐Loop Crawler with Strict “limit” on RESULTS
# -----------------------------
async def crawl_urls(job_id: str, urls: list[str], limit: int):
    """
    Crawl pages starting from `urls`, but append exactly up to `limit` results.
    Uses:
      - `seen_urls` to avoid duplicates
      - `result_count` to stop as soon as we have `limit` results
      - `count_lock` to protect both `seen_urls` and `result_count`
      - an asyncio.Queue + fixed worker coroutines
    """
    semaphore = asyncio.Semaphore(CONCURRENT_WORKERS)
    seen_urls = set()
    result_count = 0
    count_lock = asyncio.Lock()

    # 1) Seed an asyncio.Queue with the initial URLs
    queue = asyncio.Queue()
    domains = [urlparse(u).netloc for u in urls]
    for u, dom in zip(urls, domains):
        norm_top = normalize_url(u.rstrip("/"))
        if is_valid_url(norm_top, dom):
            await queue.put((norm_top, dom))

    async with httpx.AsyncClient(headers={"User-Agent": "Mozilla/5.0"}) as client:

        async def worker():
            nonlocal result_count
            while True:
                # If we've already appended `limit` results, exit immediately
                async with count_lock:
                    if result_count >= limit:
                        return

                try:
                    url, domain = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return

                # Skip if we've seen it already or if we've hit the result limit
                async with count_lock:
                    if url in seen_urls or result_count >= limit:
                        continue
                    seen_urls.add(url)

                # 2) Fetch the HTML under the semaphore
                try:
                    async with semaphore:
                        resp = await client.get(url, timeout=10.0)
                    if "text/html" not in resp.headers.get("Content-Type", ""):
                        continue
                    html = resp.text
                except Exception as e:
                    logger.warning(f"Error fetching {url}: {e}")
                    continue

                # 3) Extract the main text using trafilatura
                content = trafilatura.extract(html)
                if not content or len(content) < MIN_TEXT_LENGTH:
                    continue

                # 4) Summarize & sentiment
                summary, sentiment_dict = analyze_content(content)

                # 5) Append to results *only* if we are still under the limit
                async with count_lock:
                    if result_count >= limit:
                        return
                    result_count += 1

                await update_job(job_id, result={
                    "url": url,
                    "summary": summary,
                    **sentiment_dict
                })

                # 6) Enqueue child links only if we still can produce more results
                async with count_lock:
                    can_add_more = (result_count < limit)
                if not can_add_more:
                    return

                soup = BeautifulSoup(html, "html.parser")
                for a in soup.find_all("a", href=True):
                    full = urljoin(url, a["href"])
                    norm_child = normalize_url(full)
                    if is_valid_url(norm_child, domain):
                        async with count_lock:
                            if (norm_child not in seen_urls) and (result_count < limit):
                                await queue.put((norm_child, domain))

        # 7) Launch exactly CONCURRENT_WORKERS asynchronous workers
        workers = [asyncio.create_task(worker()) for _ in range(CONCURRENT_WORKERS)]
        await asyncio.gather(*workers)

    # 8) Once done (or limit reached), mark the job as completed
    await update_job(job_id, status="completed")

def crawl_in_background(job_id: str, urls: list[str], limit: int):
    """
    Spins up a new asyncio loop and runs `crawl_urls(...)` in this Thread.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(crawl_urls(job_id, urls, limit))
    finally:
        loop.close()

# -----------------------------
#  Flask Endpoints
# -----------------------------
@app.route("/analyse", methods=["OPTIONS", "POST"])
def analyse():
    # ---- Handle CORS preflight ----
    if request.method == "OPTIONS":
        return jsonify({}), 200

    # ---- Real POST request ----
    data = request.get_json() or {}
    urls = data.get("urls") or [data.get("url")]

    # Determine `limit` (default if not provided)
    try:
        limit = int(data.get("limit", DEFAULT_LIMIT))
    except:
        return jsonify({"error": "Invalid limit"}), 400

    # Validate each URL
    for u in urls:
        p = urlparse(u)
        if p.scheme not in ("http", "https") or not p.netloc:
            return jsonify({"error": "Invalid URL"}), 400

    # 1) Create a new job entry
    job_id = asyncio.run(create_job())

    # 2) Start the crawler in a background thread
    Thread(
        target=crawl_in_background,
        args=(job_id, urls, limit),
        daemon=True
    ).start()

    # 3) Immediately return the job_id (HTTP 202 Accepted)
    return jsonify({"job_id": job_id}), 202

@app.route("/status/<job_id>", methods=["GET"])
def status(job_id):
    job = asyncio.run(get_job(job_id))
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job), 200

# -----------------------------
#  Local Development Only
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5959)
