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
#  App initialization
# -----------------------------
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# -----------------------------
#  Configuration
# -----------------------------
MIN_TEXT_LENGTH = 200
DEFAULT_LIMIT = 10            # Default number of URLs to process if none provided
CONCURRENT_REQUESTS = 20
cache = LRUCache(maxsize=5000)

# -----------------------------
#  Sentiment Analyzer
# -----------------------------
sentiment_analyzer = SentimentIntensityAnalyzer()

# -----------------------------
#  In-memory job storage
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
#  Text processing (cached)
# -----------------------------
@cached(cache)
def analyze_content(text):
    # Summarize with TextRank (summa)
    try:
        summary = summarizer.summarize(text, ratio=0.1)
        if not summary.strip():
            raise ValueError
    except:
        summary = ". ".join(text.split(". ")[:3]) + "."
    # Sentiment scoring via vaderSentiment
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
#  Memory usage logger (optional)
# -----------------------------
def log_memory_usage():
    proc = psutil.Process()
    while True:
        mb = proc.memory_info().rss / (1024**2)
        logger.warning(f"Memory usage: {mb:.1f} MB")
        time.sleep(30)

Thread(target=log_memory_usage, daemon=True).start()

# -----------------------------
#  Helper URL functions
# -----------------------------
def normalize_url(url: str) -> str:
    url = url.split("#")[0]  # strip any fragment
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
#  Single-event-loop crawler with a URL limit
# -----------------------------
async def crawl_urls(job_id: str, urls: list[str], limit: int):
    """
    Crawl up to `limit` distinct URLs starting from `urls`.
    This runs on one asyncio loop with shared `seen` and `processed` counters.
    """
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    seen = set()
    processed = 0

    async with httpx.AsyncClient(headers={"User-Agent": "Mozilla/5.0"}) as client:

        async def fetch_and_process(url: str, domain: str):
            nonlocal processed
            # 1) If we reached the limit or this URL was already seen, stop
            if processed >= limit or url in seen:
                return
            seen.add(url)
            processed += 1

            try:
                # 2) Download HTML under semaphore
                async with semaphore:
                    resp = await client.get(url, timeout=10.0)
                if "text/html" not in resp.headers.get("Content-Type", ""):
                    return
                html = resp.text
            except Exception as e:
                logger.warning(f"Error fetching {url}: {e}")
                return

            # 3) Extract main text
            content = trafilatura.extract(html)
            if not content or len(content) < MIN_TEXT_LENGTH:
                return

            # 4) Summarize & sentiment
            summary, sentiment = analyze_content(content)
            await update_job(job_id, result={
                "url": url,
                "summary": summary,
                **sentiment
            })

            # 5) If we still can process more, enqueue children
            if processed < limit:
                soup = BeautifulSoup(html, "html.parser")
                child_tasks = []
                for a in soup.find_all("a", href=True):
                    full = urljoin(url, a["href"])
                    norm = normalize_url(full)
                    if is_valid_url(norm, domain):
                        child_tasks.append(fetch_and_process(norm, domain))
                if child_tasks:
                    await asyncio.gather(*child_tasks)

        # 6) Kick off fetch_and_process on each top-level URL
        domains = [urlparse(u).netloc for u in urls]
        top_tasks = []
        for u, dom in zip(urls, domains):
            norm_top = normalize_url(u.rstrip("/"))
            if is_valid_url(norm_top, dom):
                top_tasks.append(fetch_and_process(norm_top, dom))
        if top_tasks:
            await asyncio.gather(*top_tasks)

    # 7) Once done or limit reached, mark job completed
    await update_job(job_id, status="completed")

def crawl_in_background(job_id: str, urls: list[str], limit: int):
    """
    Create a fresh asyncio event loop in this thread and run crawl_urls(...).
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(crawl_urls(job_id, urls, limit))
    finally:
        loop.close()

# -----------------------------
#        Flask Endpoints
# -----------------------------
@app.route("/analyse", methods=["OPTIONS", "POST"])
def analyse():
    # 1) Handle CORS preflight
    if request.method == "OPTIONS":
        return jsonify({}), 200

    # 2) It's a real POST—read JSON and figure out limit/depth
    data = request.get_json() or {}
    urls = data.get("urls") or [data.get("url")]

    # Determine limit:
    if "limit" in data:
        try:
            limit = int(data["limit"])
        except:
            return jsonify({"error": "Invalid limit"}), 400
    elif "depth" in data:
        try:
            depth = int(data["depth"])
            limit = max(1, depth * 10)
        except:
            return jsonify({"error": "Invalid depth"}), 400
    else:
        limit = DEFAULT_LIMIT

    # Validate each URL
    for u in urls:
        p = urlparse(u)
        if p.scheme not in ("http", "https") or not p.netloc:
            return jsonify({"error": "Invalid URL"}), 400

    # 3) Create job_id immediately
    job_id = asyncio.run(create_job())

    # 4) Launch crawl in a background thread
    Thread(
        target=crawl_in_background,
        args=(job_id, urls, limit),
        daemon=True
    ).start()

    # 5) Return job_id at once (HTTP 202 Accepted)
    return jsonify({"job_id": job_id}), 202

@app.route("/status/<job_id>", methods=["GET"])
def status(job_id):
    job = asyncio.run(get_job(job_id))
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job), 200

# -----------------------------
#    Local development only
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5959)
