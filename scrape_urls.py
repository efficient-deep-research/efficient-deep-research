import hashlib
import json
import logging
import os
import pickle
import random
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from functools import partial
from multiprocessing import Pool

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


def fetch_url(url: str, api_key: str, cache_dir: str) -> tuple[str, requests.Response | None]:
    cache_file = os.path.join(cache_dir, hashlib.md5(url.encode("utf-8")).hexdigest() + ".pkl")
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            print("Cached: ", url)
            return url, pickle.load(f)

    try:
        response = requests.get(
            url="https://app.scrapingbee.com/api/v1",
            params={"api_key": api_key, "url": url, "render_js": "false", "return_page_text": "true"},
        )
    except Exception as e:
        logger.exception(f"Error fetching {url}: {e}")
        return url, None

    print(response.status_code, url, response.content[:100])

    with open(cache_file, "wb") as f:
        pickle.dump(response, f)

    return url, response


def main(args: Namespace) -> None:
    with open(args.input_file) as f:
        data = [json.loads(line) for line in f]

    sample_id_to_urls = defaultdict(list)
    for item in data:
        sample_id_to_urls[item["sample_id"]].append(item["url"])

    url_set = set()
    for _, urls in sorted(sample_id_to_urls.items(), key=lambda x: len(x[1]), reverse=True)[: args.top_k]:
        url_set.update(urls)

    urls = sorted(url_set)
    random.seed(1)
    random.shuffle(urls)

    os.makedirs(args.cache_dir, exist_ok=True)

    results: list[tuple[str, requests.Response | None]] = []
    func = partial(fetch_url, api_key=args.api_key, cache_dir=args.cache_dir)
    with Pool(processes=args.processes) as pool:
        with tqdm(total=len(urls)) as pbar:
            for result in pool.imap_unordered(func, urls):
                results.append(result)
                pbar.update()

    with open(args.output_file, "w") as f:
        for url, response in results:
            if response is not None and response.status_code == 200:
                item = {"url": url, "content": response.text}
                f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSONL file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSONL file.")
    parser.add_argument("--top_k", type=int, required=True, help="Number of examples having most URLs to process.")
    parser.add_argument("--api_key", type=str, required=True, help="API key for ScrapingBee.")
    parser.add_argument("--processes", type=int, default=9, help="Number of parallel processes.")
    parser.add_argument("--cache_dir", type=str, default="./cache", help="Directory to cache fetched results.")
    args = parser.parse_args()

    main(args)
