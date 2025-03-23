from flask import Flask, request, jsonify
import aiohttp
import asyncio
from bs4 import BeautifulSoup, SoupStrainer
from collections import deque
from urllib.parse import urlparse, urljoin, urldefrag
from flask_cors import CORS
from newspaper import Article
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from concurrent.futures import ProcessPoolExecutor
import logging
import torch
from cachetools import LRUCache, cached
import multiprocessing
import time
import uuid
from threading import Thread, Lock
import psutil
import time
from threading import Thread

# Initialize the Flask application
app = Flask(__name__)
CORS(app)  # Enable CORS

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Configuration constants
MIN_TEXT_LENGTH = 100
DEFAULT_DEPTH = 5
MAX_WORKERS = (multiprocessing.cpu_count() * 2) + 1

# Initialize TinyBERT model and tokenizer for sentiment analysis.
# This model is extremely lightweight and ideal for low-resource environments.
tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
model = AutoModelForSequenceClassification.from_pretrained("prajjwal1/bert-tiny", num_labels=3)

# Initialize ProcessPoolExecutor for CPU-bound tasks
executor = ProcessPoolExecutor(max_workers=MAX_WORKERS)

# Initialize Cache
cache = LRUCache(maxsize=5000)

# Helper function for timestamped logging
def log_memory_usage():
    process = psutil.Process()
    while True:
        mem_info = process.memory_info()
        print(f"Memory Usage: {mem_info.rss / (1024 * 1024):.2f} MB")  # RSS: Resident Set Size
        time.sleep(10)  # Adjust the sleep time as needed
def log(message):
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    logger.warning(f"[{current_time}] {message}")

# JobManager Class to manage crawl jobs
class JobManager:
    def __init__(self):
        self.jobs = {}
        self.lock = Lock()

    def create_job(self):
        job_id = str(uuid.uuid4())
        with self.lock:
            self.jobs[job_id] = {
                "status": "in_progress",
                "results": [],
                "error": None
            }
        return job_id

    def update_job(self, job_id, result=None, status=None, error=None):
        with self.lock:
            if job_id not in self.jobs:
                return
            if result:
                self.jobs[job_id]["results"].append(result)
            if status:
                self.jobs[job_id]["status"] = status
            if error:
                self.jobs[job_id]["error"] = error

    def get_job(self, job_id):
        with self.lock:
            return self.jobs.get(job_id, None)

# Initialize JobManager
job_manager = JobManager()

# Helper functions for URL validation and normalization
def is_valid_url(url, base_domain=None):
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ('http', 'https'):
            return False
        if not parsed.netloc:
            return False
        if base_domain and parsed.netloc != base_domain:
            return False
        if ':' in parsed.path.split('/')[-1]:
            return False
        return True
    except ValueError:
        return False

def normalize_url(url):
    url, _ = urldefrag(url)
    parsed = urlparse(url)
    normalized = parsed._replace(
        scheme=parsed.scheme.lower(),
        netloc=parsed.netloc.lower()
    )
    normalized_url = normalized.geturl().rstrip('/')
    return normalized_url

def is_relevant_text(text):
    # Ensure the text contains enough sentences
    if len(text.split('. ')) < 3:
        return False
    return True

# Asynchronous function to fetch HTML content from a URL
async def fetch(session, url):
    try:
        async with session.get(url, timeout=10) as response:
            if 'text/html' in response.headers.get('Content-Type', ''):
                text = await response.text()
                return text
    except Exception as e:
        log(f"Error fetching {url}: {e}")
    return None

# Synchronous function to fetch and parse article text using Newspaper3k
def fetch_article_text_sync(url):
    log(f"Fetching article text for URL: {url}")
    try:
        article = Article(url, fetch_images=False, request_timeout=10)
        article.download()
        article.parse()
        log(f"Successfully fetched article text for URL: {url}")
        return article.text
    except Exception as e:
        log(f"Error parsing article {url}: {e}")
        return ""

# Simple summarization: return the first few sentences as a summary
def simple_summary(text, num_sentences=3):
    sentences = text.split('. ')
    summary = '. '.join(sentences[:num_sentences])
    if summary and not summary.endswith('.'):
        summary += '.'
    return summary

# Sentiment Analysis using TinyBERT
def get_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        sentiment_score, sentiment_idx = torch.max(predictions, dim=-1)
    sentiments = ['NEGATIVE', 'NEUTRAL', 'POSITIVE']
    sentiment_label = sentiments[sentiment_idx.item()]
    star_label = "5 stars" if sentiment_label == "POSITIVE" else ("1 star" if sentiment_label == "NEGATIVE" else "3 stars")
    return {
        "star_label": star_label,
        "sentiment_label": sentiment_label,
        "sentiment_score": float(sentiment_score.item())
    }

# Combine simple summarization and sentiment analysis (with caching)
@cached(cache)
def get_summary_and_sentiment(text):
    log("Retrieving summary and sentiment from cache or computing")
    summary = simple_summary(text)
    sentiment = get_sentiment(text)
    return summary, sentiment

# Asynchronous processing of a single URL (and its linked pages)
async def process_url(session, url, method, max_pages, base_domain, processed_urls, job_id):
    pages_crawled = 0

    if method == 'bfs':
        queue = deque([url])
    else:
        queue = [url]

    while queue and pages_crawled < max_pages:
        if method == 'bfs':
            current_url = queue.popleft()
        else:
            current_url = queue.pop()

        normalized_current_url = normalize_url(current_url)

        if normalized_current_url in processed_urls:
            log(f"Skipping already processed URL within this request: {current_url}")
            continue
        processed_urls.add(normalized_current_url)

        log(f"Processing URL: {current_url}")
        html = await fetch(session, current_url)
        if html is None:
            continue

        # Extract links from the page
        links = BeautifulSoup(html, "html.parser", parse_only=SoupStrainer('a'))
        for link in links.find_all('a', href=True):
            full_url = urljoin(current_url, link['href'])
            normalized_full_url = normalize_url(full_url)
            if is_valid_url(full_url, base_domain=base_domain) and normalized_full_url not in processed_urls:
                queue.append(full_url)

        # Fetch article text synchronously via executor
        page_text = await asyncio.get_event_loop().run_in_executor(executor, fetch_article_text_sync, current_url)
        if len(page_text.strip()) < MIN_TEXT_LENGTH:
            log(f"Skipped {current_url} due to insufficient content.")
            continue

        if not is_relevant_text(page_text):
            log(f"Skipped {current_url} due to irrelevant content.")
            continue

        summary, sentiment = get_summary_and_sentiment(page_text)

        if not summary:
            log(f"Skipped summarization for {current_url} due to empty summary.")
            continue

        result = {
            "url": current_url,
            "summary": summary,
            "star_label": sentiment["star_label"],
            "sentiment_label": sentiment["sentiment_label"],
            "sentiment_score": sentiment["sentiment_score"]
        }
        job_manager.update_job(job_id, result=result)

        pages_crawled += 1
        log(f"Completed processing URL: {current_url}")

# Endpoint to check the status of a crawl job
@app.route('/status/<job_id>', methods=['GET'])
def get_status(job_id):
    job = job_manager.get_job(job_id)
    if not job:
        return jsonify({"error": "Job ID not found."}), 404

    response = {
        "status": job["status"],
        "results": job["results"],
        "error": job["error"]
    }
    return jsonify(response), 200

# Asynchronous function to run crawl tasks for given URLs
async def run_crawl_async(job_id, urls, method, depth, base_domains):
    async with aiohttp.ClientSession(headers={"User-Agent": "Mozilla/5.0"}) as session:
        tasks = []
        for url, base_domain in zip(urls, base_domains):
            processed_urls = set()
            tasks.append(process_url(session, url, method, depth, base_domain, processed_urls, job_id))
        await asyncio.gather(*tasks)

# Wrapper to run asynchronous crawl tasks in a separate event loop
def run_crawl(job_id, urls, method, depth, base_domains):
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(run_crawl_async(job_id, urls, method, depth, base_domains))
        loop.close()
        job_manager.update_job(job_id, status="completed")
        log(f"Completed crawl job {job_id}")
    except Exception as e:
        log(f"Error in crawl job {job_id}: {e}")
        job_manager.update_job(job_id, status="failed", error=str(e))

# Endpoint to start the analysis/crawl process
@app.route('/analyse', methods=['POST'])
def analyse():
    data = request.get_json()
    urls = data.get('urls', [data.get('url')])
    method = data.get('method', 'bfs').lower()
    depth = data.get('depth', DEFAULT_DEPTH)

    if not urls or not all(is_valid_url(url) for url in urls):
        log("Invalid URL or URLs provided.")
        return jsonify({"error": "A valid 'url' or 'urls' is required."}), 400
    if method not in ['bfs', 'dfs']:
        log("Invalid method provided.")
        return jsonify({"error": "Method must be 'bfs' or 'dfs'."}), 400

    job_id = job_manager.create_job()
    base_domains = [urlparse(url).netloc for url in urls]

    thread = Thread(target=run_crawl, args=(job_id, urls, method, depth, base_domains))
    thread.start()

    log(f"Started crawl job {job_id}")
    return jsonify({"job_id": job_id}), 202
memory_thread = Thread(target=log_memory_usage)
memory_thread.start()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5959, debug=False)
