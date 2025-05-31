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

# Allow CORS on every route (wildcard origin)
CORS(app, resources={r"/*": {"origins": "*"}})

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# -----------------------------
#     Configuration
# -----------------------------
MIN_TEXT_LENGTH = 200
DEFAULT_LIMIT = 10         # Default # of results if none provided
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
    1) Extract an extractive summary using summa.TextRank (fallback to first 3 sentences).
    2) Run VADER to get a compound score in [-1.0, +1.0].
    3) Confidence = |compound| ∈ [0.0, 1.0].
    4) Star rating = round(((compound + 1)/2)*4) + 1  → integer in {1,2,3,4,5}.
    5) Sentiment label = POSITIVE/NEGATIVE/NEUTRAL.
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
    comp = scores["compound"]  # in [-1.0 … +1.0]

    # ---- Confidence = absolute(compound) in [0.0 … 1.0] ----
    confidence = abs(comp)

    # ---- Star rating = discrete 1..5, linear map from [-1..+1] → [1..5] ----
    #   formula: star_num = round(((comp + 1.0)/2.0)*4.0) + 1
    star_num = int(round(((comp + 1.0) / 2.0) * 4.0)) + 1
    # Just in case numerical rounding goes outside 1..5:
    if star_num < 1:
        star_num = 1
    elif star_num > 5:
        star_num = 5
    star_label = f"{star_num} stars"

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
        # Round confidence to 3 decimal places for JSON clarity
        "sentiment_score": round(confidence, 3)
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
async def crawl_urls(job_id: str, seed_urls: list[str], limit: int):
    """
    Perform a breadth‐first traversal (no parallelism) that:
      - Visits each URL from a FIFO queue.
      - Extracts content only if it’s valid text/html and ≥ MIN_TEXT_LENGTH.
      - Appends exactly `limit` results to jobs[job_id]["results"].
      - Stops immediately once we've gathered `limit` results.
      - Enqueues children only until `limit` is reached.
    """
    seen_urls = set()
    result_count = 0

    from collections import deque
    queue = deque()
    domains = [urlparse(u).netloc for u in seed_urls]

    # 1) Seed the queue with normalized seed URLs
    for u, dom in zip(seed_urls, domains):
        norm_top = normalize_url(u.rstrip("/"))
        if is_valid_url(norm_top, dom):
            queue.append((norm_top, dom))

    async with httpx.AsyncClient(headers={"User-Agent": "Mozilla/5.0"}) as client:
        while queue and result_count < limit:
            url, domain = queue.popleft()

            # Skip if already seen
            if url in seen_urls:
                continue
            seen_urls.add(url)

            # 2) Fetch HTML
            try:
                resp = await client.get(url, timeout=10.0)
                if "text/html" not in resp.headers.get("Content-Type", ""):
                    continue
                html = resp.text
            except Exception as e:
                logger.warning(f"Error fetching {url}: {e}")
                continue

            # 3) Extract main text
            content = trafilatura.extract(html)
            if not content or len(content) < MIN_TEXT_LENGTH:
                continue

            # 4) Summarize & sentiment
            summary, sentiment_dict = analyze_content(content)

            # 5) Append exactly one result item if still under the limit
            if result_count < limit:
                result_count += 1
                await update_job(job_id, result={
                    "url": url,
                    "summary": summary,
                    **sentiment_dict
                })

            # 6) If we have now reached the limit, break out
            if result_count >= limit:
                break

            # 7) Enqueue children (links) so long as we haven't hit `limit`
            soup = BeautifulSoup(html, "html.parser")
            for a in soup.find_all("a", href=True):
                full = urljoin(url, a["href"])
                norm_child = normalize_url(full)
                if is_valid_url(norm_child, domain) and norm_child not in seen_urls:
                    queue.append((norm_child, domain))

    # 8) Mark job as completed
    await update_job(job_id, status="completed")

def crawl_in_background(job_id: str, urls: list[str], limit: int):
    """
    Spins up a new asyncio event loop in this thread and calls `crawl_urls(...)`.
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
