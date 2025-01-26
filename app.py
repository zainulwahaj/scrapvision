from flask import Flask, request, jsonify
import aiohttp
import asyncio
from bs4 import BeautifulSoup, SoupStrainer
from collections import deque
from urllib.parse import urlparse, urljoin, urldefrag
from flask_cors import CORS
from newspaper import Article
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration
from concurrent.futures import ThreadPoolExecutor
import nest_asyncio
import logging
import torch
from cachetools import LRUCache, cached
import multiprocessing
import time
import uuid
from threading import Thread, Lock

# Apply nest_asyncio to allow asyncio within Flask
nest_asyncio.apply()

# Initialize the Flask application
app = Flask(__name__)
CORS(app)  # Enable CORS

# Configure logging
logging.basicConfig(level=logging.WARNING)  # Reduced logging level to WARNING to minimize overhead
logger = logging.getLogger(__name__)

# Configuration constants
MIN_TEXT_LENGTH = 100
MAX_T5_INPUT_TOKENS = 512  # Adjusted for t5-small
T5_NUM_BEAMS = 2
DEFAULT_DEPTH = 5
MAX_WORKERS = (multiprocessing.cpu_count() * 2) + 1  # Optimized based on CPU cores

# Initialize models and tokenizer
# Summarization using T5-small
t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')
t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')

# Sentiment analysis using nlptown's BERT model (1-5 stars)
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment"
)

# Initialize ThreadPoolExecutor for CPU-bound tasks
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# Initialize Cache
cache = LRUCache(maxsize=5000)  # Adjust based on your needs

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
    """
    Validates the URL.
    - Must have http or https scheme.
    - Must belong to the base_domain if provided.
    - Exclude specific paths like 'Main_Page' or any non-article namespaces.
    """
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ('http', 'https'):
            return False
        if not parsed.netloc:
            return False
        if base_domain and parsed.netloc != base_domain:
            return False
        # Exclude URLs with ':' in the last path segment to skip non-article namespaces (e.g., 'Wikipedia:Contents')
        if ':' in parsed.path.split('/')[-1]:
            return False
        return True
    except ValueError:
        return False

def normalize_url(url):
    """
    Normalizes the URL by:
    - Removing fragment identifiers.
    - Lowercasing the scheme and netloc.
    - Stripping trailing slashes.
    """
    url, _ = urldefrag(url)  # Remove fragment
    parsed = urlparse(url)
    normalized = parsed._replace(
        scheme=parsed.scheme.lower(),
        netloc=parsed.netloc.lower()
    )
    normalized_url = normalized.geturl().rstrip('/')
    return normalized_url

def is_relevant_text(text):
    """
    Determines if the extracted text is relevant for summarization and sentiment analysis.
    """
    # Heuristic: Check for multiple sentences
    if len(text.split('. ')) < 3:
        return False
    return True

async def fetch(session, url):
    """
    Asynchronously fetches the content of the URL.
    """
    try:
        async with session.get(url, timeout=10) as response:
            if 'text/html' in response.headers.get('Content-Type', ''):
                text = await response.text()
                return text
    except Exception as e:
        log(f"Error fetching {url}: {e}")
    return None

def fetch_article_text_sync(url):
    """
    Synchronously fetches and parses the article text from the URL.
    """
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

def summarize_text_sync(text):
    """
    Synchronously summarizes the provided text using T5-small.
    """
    log("Starting summarization")
    try:
        input_text = "summarize: " + text
        inputs = t5_tokenizer.encode(
            input_text,
            return_tensors="pt",
            max_length=MAX_T5_INPUT_TOKENS,
            truncation=True
        )
        summary_ids = t5_model.generate(
            inputs,
            max_length=150,
            min_length=40,
            length_penalty=2.0,
            num_beams=T5_NUM_BEAMS,
            early_stopping=True
        )
        summary = t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        log("Summarization completed")
        return summary
    except Exception as e:
        log(f"Error summarizing text: {e}")
        return ""

def analyze_sentiment_sync(text):
    """
    Synchronously analyzes the sentiment of the provided text.
    Maps the model's output to a star rating.
    """
    log("Starting sentiment analysis")
    try:
        # The nlptown model outputs labels like '1 star', '2 stars', ..., '5 stars'
        result = sentiment_analyzer(text, truncation=True, max_length=512)[0]
        label = result['label']
        score = result['score']

        # Ensure the label is in the expected format
        if label in ['1 star', '2 stars', '3 stars', '4 stars', '5 stars']:
            sentiment_label = label
            # Optionally, you can map these to 'NEGATIVE', 'NEUTRAL', 'POSITIVE' if needed
            if label in ['1 star', '2 stars']:
                overall_sentiment = "NEGATIVE"
            elif label == '3 stars':
                overall_sentiment = "NEUTRAL"
            else:
                overall_sentiment = "POSITIVE"
        else:
            # Fallback in case of unexpected label
            sentiment_label = "3 stars"
            overall_sentiment = "NEUTRAL"
            score = 0.0

        log("Sentiment analysis completed")
        return {
            "star_label": sentiment_label,
            "sentiment_label": overall_sentiment,
            "sentiment_score": score
        }
    except Exception as e:
        log(f"Error analyzing sentiment: {e}")
        return {
            "star_label": "3 stars",
            "sentiment_label": "NEUTRAL",
            "sentiment_score": 0.0
        }

@cached(cache)
def get_summary_and_sentiment(text):
    """
    Retrieves the summary and sentiment from the cache or computes them if not cached.
    """
    log("Retrieving summary and sentiment from cache or computing")
    summary = summarize_text_sync(text)
    sentiment = analyze_sentiment_sync(text)
    return summary, sentiment

async def process_url(session, url, method, max_pages, base_domain, processed_urls, job_id):
    """
    Processes a single URL by fetching, extracting, summarizing, and analyzing sentiment.
    Tracks processed URLs within the scope of this request.
    """
    pages_crawled = 0

    if method == 'bfs':
        queue = deque([url])
    else:  # dfs
        queue = [url]

    while queue and pages_crawled < max_pages:
        if method == 'bfs':
            current_url = queue.popleft()
        else:
            current_url = queue.pop()

        normalized_current_url = normalize_url(current_url)

        # Check if URL has already been processed within this request
        if normalized_current_url in processed_urls:
            log(f"Skipping already processed URL within this request: {current_url}")
            continue
        # Mark as processed within this request
        processed_urls.add(normalized_current_url)

        log(f"Processing URL: {current_url}")
        html = await fetch(session, current_url)
        if html is None:
            continue

        # Parse links
        links = BeautifulSoup(html, "html.parser", parse_only=SoupStrainer('a'))
        for link in links.find_all('a', href=True):
            full_url = urljoin(current_url, link['href'])
            normalized_full_url = normalize_url(full_url)
            if is_valid_url(full_url, base_domain=base_domain) and normalized_full_url not in processed_urls:
                queue.append(full_url)

        # Extract and process article text
        loop = asyncio.get_event_loop()
        page_text = await loop.run_in_executor(executor, fetch_article_text_sync, current_url)
        if len(page_text.strip()) < MIN_TEXT_LENGTH:
            log(f"Skipped {current_url} due to insufficient content.")
            continue

        if not is_relevant_text(page_text):
            log(f"Skipped {current_url} due to irrelevant content.")
            continue

        # Retrieve from cache or compute summary and sentiment
        summary, sentiment = get_summary_and_sentiment(page_text)

        if not summary:
            log(f"Skipped summarization for {current_url} due to empty summary.")
            continue

        # Update the job with the new result
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
    """
    Returns the current status and results of the specified job.
    """
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
            # Initialize a set to track processed URLs for this request
            processed_urls = set()
            tasks.append(process_url(session, url, method, depth, base_domain, processed_urls, job_id))
        await asyncio.gather(*tasks)

def run_crawl(job_id, urls, method, depth, base_domains):
    """
    Runs the crawl process and updates the job status and results.
    """
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
    """
    Flask route to handle analysis requests.
    Expects JSON with 'url' or 'urls', 'method', and 'depth'.
    Returns a unique job_id.
    """
    data = request.get_json()
    urls = data.get('urls', [data.get('url')])  # Support batch or single URL
    method = data.get('method', 'bfs').lower()
    depth = data.get('depth', DEFAULT_DEPTH)

    if not urls or not all(is_valid_url(url) for url in urls):
        log("Invalid URL or URLs provided.")
        return jsonify({"error": "A valid 'url' or 'urls' is required."}), 400
    if method not in ['bfs', 'dfs']:
        log("Invalid method provided.")
        return jsonify({"error": "Method must be 'bfs' or 'dfs'."}), 400

    # Create a new job
    job_id = job_manager.create_job()

    # Determine the base domain(s) to restrict crawling within the same domain
    base_domains = [urlparse(url).netloc for url in urls]

    # Start the crawl in a background thread
    thread = Thread(target=run_crawl, args=(job_id, urls, method, depth, base_domains))
    thread.start()

    log(f"Started crawl job {job_id}")
    return jsonify({"job_id": job_id}), 202  # 202 Accepted

# Main entry to run the application
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5959, debug=False)