from flask import Flask, request, jsonify
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

# -- App setup --
app = Flask(__name__)
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# -- Configuration --
MIN_TEXT_LENGTH = 200
DEFAULT_DEPTH = 2
CONCURRENT_REQUESTS = 20
cache = LRUCache(maxsize=5000)

# -- Sentiment analyzer (vaderSentiment, no external download needed) --
sentiment_analyzer = SentimentIntensityAnalyzer()

# -- Job storage --
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

# -- Text processing --
@cached(cache)
def analyze_content(text):
    # Summarize with TextRank
    try:
        summary = summarizer.summarize(text, ratio=0.1)
        if not summary.strip():
            raise ValueError
    except:
        summary = ". ".join(text.split(". ")[:3]) + "."
    # Sentiment scoring
    scores = sentiment_analyzer.polarity_scores(text)
    comp = scores["compound"]
    if comp >= 0.05:
        star, label = "5 stars", "POSITIVE"
    elif comp <= -0.05:
        star, label = "1 star", "NEGATIVE"
    else:
        star, label = "3 stars", "NEUTRAL"
    return summary, {"star_label": star, "sentiment_label": label, "sentiment_score": comp}

# -- Memory logging --
def log_memory():
    proc = psutil.Process()
    while True:
        mb = proc.memory_info().rss / (1024**2)
        logger.warning(f"Memory usage: {mb:.1f} MB")
        time.sleep(30)

Thread(target=log_memory, daemon=True).start()

# -- Async crawl --
semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

def normalize_url(url):
    url = url.split('#')[0]
    p = urlparse(url)
    norm = p._replace(scheme=p.scheme.lower(), netloc=p.netloc.lower())
    return norm.geturl().rstrip('/')

def is_valid_url(url, domain=None):
    try:
        p = urlparse(url)
        if p.scheme not in ("http", "https") or not p.netloc:
            return False
        if domain and p.netloc != domain:
            return False
        return True
    except:
        return False

async def fetch_and_process(client, url, depth, domain, seen, job_id):
    if depth < 0 or url in seen:
        return
    seen.add(url)
    try:
        async with semaphore:
            resp = await client.get(url, timeout=10.0)
        if "text/html" not in resp.headers.get("Content-Type", ""):
            return
        html = resp.text
    except Exception as e:
        logger.warning(f"Error fetching {url}: {e}")
        return

    # extract main text
    content = trafilatura.extract(html)
    if not content or len(content) < MIN_TEXT_LENGTH:
        return

    summary, sentiment = analyze_content(content)
    await update_job(job_id, result={
        "url": url,
        "summary": summary,
        **sentiment
    })

    # parse links with BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.find_all("a", href=True):
        full = urljoin(url, a["href"])
        norm = normalize_url(full)
        if is_valid_url(norm, domain):
            asyncio.create_task(fetch_and_process(client, norm, depth-1, domain, seen, job_id))

@app.route("/analyse", methods=["POST"])
def analyse():
    data = request.get_json() or {}
    urls = data.get("urls") or [data.get("url")]
    depth = data.get("depth", DEFAULT_DEPTH)

    # validate
    for u in urls:
        p = urlparse(u)
        if p.scheme not in ("http", "https") or not p.netloc:
            return jsonify({"error": "Invalid URL"}), 400

    async def runner():
        job_id = await create_job()
        domains = [urlparse(u).netloc for u in urls]
        async with httpx.AsyncClient(headers={"User-Agent": "Mozilla/5.0"}) as client:
            tasks = [
                fetch_and_process(client, u.rstrip("/"), depth, dom, set(), job_id)
                for u, dom in zip(urls, domains)
            ]
            await asyncio.gather(*tasks)
        await update_job(job_id, status="completed")
        return job_id

    job_id = asyncio.run(runner())
    return jsonify({"job_id": job_id}), 202

@app.route("/status/<job_id>", methods=["GET"])
def status(job_id):
    job = asyncio.run(get_job(job_id))
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5959)
