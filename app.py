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

# -----------------------------
#     Logging Configuration
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("scrapvision")

# -----------------------------
#     Configuration
# -----------------------------
MIN_TEXT_LENGTH = 200
DEFAULT_LIMIT = 10            # Default # of results if none provided
CONCURRENT_WORKERS = 10       # Number of concurrent worker tasks
MAX_SUMMARY_WORDS = 30        # Maximum number of words in each summary
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
    1) Summarize with summa.TextRank (fallback to first MAX_SUMMARY_WORDS words).
    2) Sentiment via VADER → compound in [-1,1] → map to confidence ∈ [0.0,1.0].
    3) Map that confidence to star buckets (1..5).
    """
    summary = ""
    try:
        # ---- Summarization (TextRank via summa) ----
        summary = summarizer.summarize(text, ratio=0.1)
        if not summary.strip():
            raise ValueError("summa returned empty summary")
    except Exception as e:
        # Fallback: first few sentences up to MAX_SUMMARY_WORDS words
        logger.debug(f"Summa failed or returned empty, falling back: {e}")
        sentences = text.split(". ")
        fallback = " ".join(sentences[:3]) + "."
        summary = fallback

    # Trim summary to MAX_SUMMARY_WORDS words
    words = summary.split()
    if len(words) > MAX_SUMMARY_WORDS:
        words = words[:MAX_SUMMARY_WORDS]
        words[-1] = words[-1].rstrip(".,;:!?") + "…"  # add ellipsis to last word
    summary = " ".join(words)

    # ---- Sentiment via VADER ----
    scores = sentiment_analyzer.polarity_scores(text)
    comp = scores["compound"]  # in [-1.0 … +1.0]

    # ---- Confidence = absolute(compound) in [0.0 … 1.0] ----
    confidence = abs(comp)

    # ---- Star rating = discrete 1..5, linear map from [-1..+1] → [1..5] ----
    #    formula: star_num = round(((comp + 1.0)/2.0)*4.0) + 1
    star_num = int(round(((comp + 1.0) / 2.0) * 4.0)) + 1
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
        "sentiment_score": round(confidence, 3),
    }

# -----------------------------
#  Memory Usage Logger (optional)
# -----------------------------
def log_memory_usage():
    proc = psutil.Process()
    while True:
        mb = proc.memory_info().rss / (1024**2)
        logger.info(f"Memory usage: {mb:.1f} MB")
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
    except Exception:
        return False

# -----------------------------
#  Single‐Loop Crawler with Strict “limit” on RESULTS
# -----------------------------
async def crawl_urls(job_id: str, seed_urls: list[str], limit: int):
    """
    Perform a breadth‐first traversal of `seed_urls`, but append exactly up to `limit` results.
    Stops immediately once `limit` results are stored.
    """
    try:
        from collections import deque
        seen_urls = set()
        result_count = 0
        queue = deque()
        domains = [urlparse(u).netloc for u in seed_urls]

        # 1) Seed the queue
        for u, dom in zip(seed_urls, domains):
            norm_top = normalize_url(u.rstrip("/"))
            if is_valid_url(norm_top, dom):
                queue.append((norm_top, dom))

        async with httpx.AsyncClient(headers={"User-Agent": "Mozilla/5.0"}) as client:
            while queue and result_count < limit:
                url, domain = queue.popleft()

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
                    logger.warning(f"[crawl_urls] Failed to fetch {url}: {e}")
                    continue

                # 3) Extract main text
                content = trafilatura.extract(html)
                if not content or len(content) < MIN_TEXT_LENGTH:
                    continue

                # 4) Summarize & sentiment
                try:
                    summary, sentiment_dict = analyze_content(content)
                except Exception as e:
                    logger.error(f"[crawl_urls] analyze_content error for {url}: {e}", exc_info=True)
                    continue

                # 5) Append if still under limit
                if result_count < limit:
                    result_count += 1
                    await update_job(job_id, result={
                        "url": url,
                        "summary": summary,
                        **sentiment_dict
                    })

                # 6) Stop if limit reached
                if result_count >= limit:
                    break

                # 7) Enqueue children
                soup = BeautifulSoup(html, "html.parser")
                for a in soup.find_all("a", href=True):
                    full = urljoin(url, a["href"])
                    norm_child = normalize_url(full)
                    if is_valid_url(norm_child, domain) and norm_child not in seen_urls:
                        queue.append((norm_child, domain))

        # 8) Mark job completed
        await update_job(job_id, status="completed")

    except Exception as e:
        # Catch any unexpected exception, log it, and mark job as failed
        logger.error(f"[crawl_urls] Unexpected error: {e}", exc_info=True)
        await update_job(job_id, status="failed", error=str(e))

def crawl_in_background(job_id: str, urls: list[str], limit: int):
    """
    Spins up a new asyncio loop in this thread and runs `crawl_urls`.
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

    try:
        data = request.get_json() or {}
        urls = data.get("urls") or [data.get("url")]

        # Determine `limit`
        try:
            limit = int(data.get("limit", DEFAULT_LIMIT))
            if limit < 1:
                raise ValueError("limit must be ≥ 1")
        except Exception as e:
            return jsonify({"error": f"Invalid limit: {e}"}), 400

        # Validate each URL
        for u in urls:
            p = urlparse(u)
            if p.scheme not in ("http", "https") or not p.netloc:
                return jsonify({"error": f"Invalid URL: {u}"}), 400

        # Create a new job
        job_id = asyncio.run(create_job())

        # Launch background crawler
        Thread(
            target=crawl_in_background,
            args=(job_id, urls, limit),
            daemon=True
        ).start()

        return jsonify({"job_id": job_id}), 202

    except Exception as e:
        logger.error(f"[endpoint /analyse] unexpected error: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

@app.route("/status/<job_id>", methods=["GET"])
def status(job_id):
    try:
        job = asyncio.run(get_job(job_id))
        if not job:
            return jsonify({"error": "Job not found"}), 404
        return jsonify(job), 200
    except Exception as e:
        logger.error(f"[endpoint /status] unexpected error: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

# -----------------------------
#  Local Development Only
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5959)
