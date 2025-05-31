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
CORS(app)  # Enable CORS for all routes
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# -----------------------------
#     Configuration
# -----------------------------
MIN_TEXT_LENGTH = 200
DEFAULT_LIMIT = 10         # Default # of URLs to scrape if none passed
CONCURRENT_WORKERS = 10    # Number of concurrent workers pulling from the queue
cache = LRUCache(maxsize=5000)

# -----------------------------
#  Sentiment Analyzer (bundled)
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
def analyze_content(text):
    # 1) Summarize (TextRank via summa)
    try:
        summary = summarizer.summarize(text, ratio=0.1)
        if not summary.strip():
            raise ValueError
    except:
        summary = ". ".join(text.split(". ")[:3]) + "."

    # 2) Sentiment via vaderSentiment
    scores = sentiment_analyzer.polarity_scores(text)
    comp = scores["compound"]
    if comp >= 0.05:
        star, label = "5 stars", "POSITIVE"
    elif comp <= -0.05:
        star, label = "1 star", "NEGATIVE"
    else:
        star, label = "3 stars", "NEUTRAL"

    return summary, {"star_label": star, "sentiment_label": label, "sentiment_score": comp}

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
    # Strip fragment, lowercase scheme & host
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
#  Single‐Loop Crawler with Strict URL Limit
# -----------------------------
async def crawl_urls(job_id: str, urls: list[str], limit: int):
    """
    Crawl up to `limit` distinct URLs starting from `urls`.
    Uses an asyncio.Queue + worker tasks:
      - A shared `seen` set of URLs already fetched
      - A shared `processed` counter (protected by `count_lock`)
      - A shared semaphore for HTTP concurrency
      - Exactly `limit` URLs will be analyzed at most
    """
    semaphore = asyncio.Semaphore(CONCURRENT_WORKERS)
    seen = set()
    processed = 0
    count_lock = asyncio.Lock()

    # 1) Create an asyncio.Queue and seed with the initial URLs
    queue = asyncio.Queue()
    domains = [urlparse(u).netloc for u in urls]
    for u, dom in zip(urls, domains):
        norm_top = normalize_url(u.rstrip("/"))
        if is_valid_url(norm_top, dom):
            await queue.put((norm_top, dom))

    async with httpx.AsyncClient(headers={"User-Agent": "Mozilla/5.0"}) as client:

        async def worker():
            nonlocal processed
            while True:
                # If we've already processed `limit`, exit
                async with count_lock:
                    if processed >= limit:
                        return

                try:
                    url, domain = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return

                # Skip if we've seen it already
                async with count_lock:
                    if url in seen or processed >= limit:
                        continue
                    seen.add(url)
                    processed += 1  # count this URL toward the limit

                # 2) Fetch HTML under the semaphore
                try:
                    async with semaphore:
                        resp = await client.get(url, timeout=10.0)
                    if "text/html" not in resp.headers.get("Content-Type", ""):
                        continue
                    html = resp.text
                except Exception as e:
                    logger.warning(f"Error fetching {url}: {e}")
                    continue

                # 3) Extract main text with trafilatura
                content = trafilatura.extract(html)
                if not content or len(content) < MIN_TEXT_LENGTH:
                    continue

                # 4) Summarize & sentiment
                summary, sentiment = analyze_content(content)
                await update_job(job_id, result={
                    "url": url,
                    "summary": summary,
                    **sentiment
                })

                # 5) Enqueue children only if we still can process more
                async with count_lock:
                    can_continue = (processed < limit)
                if not can_continue:
                    return

                soup = BeautifulSoup(html, "html.parser")
                for a in soup.find_all("a", href=True):
                    full = urljoin(url, a["href"])
                    norm_child = normalize_url(full)
                    if is_valid_url(norm_child, domain):
                        async with count_lock:
                            # Only put child in queue if not already seen AND we haven't hit limit
                            if (norm_child not in seen) and (processed < limit):
                                await queue.put((norm_child, domain))

        # 6) Launch a fixed number of concurrent worker tasks
        workers = [asyncio.create_task(worker()) for _ in range(CONCURRENT_WORKERS)]
        await asyncio.gather(*workers)

    # 7) Once done or limit reached, mark job as completed
    await update_job(job_id, status="completed")

def crawl_in_background(job_id: str, urls: list[str], limit: int):
    """
    Create a dedicated asyncio event loop in this thread and run `crawl_urls(...)`.
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
