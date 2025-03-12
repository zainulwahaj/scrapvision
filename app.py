from flask import Flask, request, jsonify
import aiohttp
import asyncio
from bs4 import BeautifulSoup, SoupStrainer
from collections import deque
from urllib.parse import urlparse, urljoin, urldefrag
from flask_cors import CORS
from newspaper import Article
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from concurrent.futures import ProcessPoolExecutor
import logging
import torch
from cachetools import LRUCache, cached
import multiprocessing
import time
import uuid
from threading import Thread, Lock

# Initialize the Flask application
app = Flask(__name__)
CORS(app)  # Enable CORS

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Configuration constants
MIN_TEXT_LENGTH = 100
MAX_GPT_INPUT_TOKENS = 512
DEFAULT_DEPTH = 5
MAX_WORKERS = (multiprocessing.cpu_count() * 2) + 1

# Initialize GPT-Neo model and tokenizer
gpt_neo_model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
gpt_neo_tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

# Initialize ProcessPoolExecutor for CPU-bound tasks
executor = ProcessPoolExecutor(max_workers=MAX_WORKERS)

# Initialize Cache
cache = LRUCache(maxsize=5000)

# Helper function for timestamped logging
def log(message):
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    logger.warning(f"[{current_time}] {message}")

# JobManager Class
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

# Helper functions
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
    if len(text.split('. ')) < 3:
        return False
    return True

async def fetch(session, url):
    try:
        async with session.get(url, timeout=10) as response:
            if 'text/html' in response.headers.get('Content-Type', ''):
                text = await response.text()
                return text
    except Exception as e:
        log(f"Error fetching {url}: {e}")
    return None

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

def summarize_text_with_gpt_neo(text, max_length=150):
    input_text = f"Summarize the following text:\n{text}\n\nSummary:"
    inputs = gpt_neo_tokenizer(input_text, return_tensors="pt", max_length=MAX_GPT_INPUT_TOKENS, truncation=True)
    summary_ids = gpt_neo_model.generate(
        inputs.input_ids,
        max_length=max_length,
        num_beams=2,
        early_stopping=True
    )
    summary = gpt_neo_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def analyze_sentiment_with_gpt_neo(text):
    input_text = f"Analyze the sentiment of the following text and respond with 'POSITIVE', 'NEUTRAL', or 'NEGATIVE':\n{text}\n\nSentiment:"
    inputs = gpt_neo_tokenizer(input_text, return_tensors="pt", max_length=MAX_GPT_INPUT_TOKENS, truncation=True)
    output_ids = gpt_neo_model.generate(
        inputs.input_ids,
        max_length=10,
        num_beams=1,
        early_stopping=True
    )
    sentiment = gpt_neo_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    if "POSITIVE" in sentiment:
        return {"star_label": "5 stars", "sentiment_label": "POSITIVE", "sentiment_score": 1.0}
    elif "NEGATIVE" in sentiment:
        return {"star_label": "1 star", "sentiment_label": "NEGATIVE", "sentiment_score": 0.0}
    else:
        return {"star_label": "3 stars", "sentiment_label": "NEUTRAL", "sentiment_score": 0.5}

@cached(cache)
def get_summary_and_sentiment(text):
    log("Retrieving summary and sentiment from cache or computing")
    summary = summarize_text_with_gpt_neo(text)
    sentiment = analyze_sentiment_with_gpt_neo(text)
    return summary, sentiment

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

        links = BeautifulSoup(html, "html.parser", parse_only=SoupStrainer('a'))
        for link in links.find_all('a', href=True):
            full_url = urljoin(current_url, link['href'])
            normalized_full_url = normalize_url(full_url)
            if is_valid_url(full_url, base_domain=base_domain) and normalized_full_url not in processed_urls:
                queue.append(full_url)

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

async def run_crawl_async(job_id, urls, method, depth, base_domains):
    async with aiohttp.ClientSession(headers={"User-Agent": "Mozilla/5.0"}) as session:
        tasks = []
        for url, base_domain in zip(urls, base_domains):
            processed_urls = set()
            tasks.append(process_url(session, url, method, depth, base_domain, processed_urls, job_id))
        await asyncio.gather(*tasks)

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5959, debug=False)