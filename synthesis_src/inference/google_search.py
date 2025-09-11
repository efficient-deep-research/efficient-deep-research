import os
import json
import requests
from requests.exceptions import Timeout
from bs4 import BeautifulSoup
from tqdm import tqdm
import time
import concurrent
from concurrent.futures import ThreadPoolExecutor
import pdfplumber
from io import BytesIO
import re
import string
from typing import Optional, Tuple
from nltk.tokenize import sent_tokenize

# os.environ['http_proxy'] = 'http://127.0.0.1:7890'
# os.environ['https_proxy'] = 'http://127.0.0.1:7890'


# ----------------------- Custom Headers -----------------------
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/58.0.3029.110 Safari/537.36',
    'Referer': 'https://www.google.com/',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1'
}

# Initialize session
session = requests.Session()
session.headers.update(headers)



def remove_punctuation(text: str) -> str:
    """Remove punctuation from the text."""
    return text.translate(str.maketrans("", "", string.punctuation))

def f1_score(true_set: set, pred_set: set) -> float:
    """Calculate the F1 score between two sets of words."""
    intersection = len(true_set.intersection(pred_set))
    if not intersection:
        return 0.0
    precision = intersection / float(len(pred_set))
    recall = intersection / float(len(true_set))
    return 2 * (precision * recall) / (precision + recall)

def extract_snippet_with_context(full_text: str, snippet: str, context_chars: int = 2500) -> Tuple[bool, str]:
    """
    Extract the sentence that best matches the snippet and its context from the full text.

    Args:
        full_text (str): The full text extracted from the webpage.
        snippet (str): The snippet to match.
        context_chars (int): Number of characters to include before and after the snippet.

    Returns:
        Tuple[bool, str]: The first element indicates whether extraction was successful, the second element is the extracted context.
    """
    try:
        full_text = full_text[:50000]

        snippet = snippet.lower()
        snippet = remove_punctuation(snippet)
        snippet_words = set(snippet.split())

        best_sentence = None
        best_f1 = 0.2

        # sentences = re.split(r'(?<=[.!?]) +', full_text)  # Split sentences using regex, supporting ., !, ? endings
        sentences = sent_tokenize(full_text)  # Split sentences using nltk's sent_tokenize

        for sentence in sentences:
            key_sentence = sentence.lower()
            key_sentence = remove_punctuation(key_sentence)
            sentence_words = set(key_sentence.split())
            f1 = f1_score(snippet_words, sentence_words)
            if f1 > best_f1:
                best_f1 = f1
                best_sentence = sentence

        if best_sentence:
            para_start = full_text.find(best_sentence)
            para_end = para_start + len(best_sentence)
            start_index = max(0, para_start - context_chars)
            end_index = min(len(full_text), para_end + context_chars)
            context = full_text[start_index:end_index]
            return True, context
        else:
            # If no matching sentence is found, return the first context_chars*2 characters of the full text
            return False, full_text[:context_chars * 2]
    except Exception as e:
        return False, f"Failed to extract snippet context due to {str(e)}"

def extract_text_from_url(url, use_jina=False, jina_api_key=None, snippet: Optional[str] = None):
    """
    Extract text from a URL. If a snippet is provided, extract the context related to it.

    Args:
        url (str): URL of a webpage or PDF.
        use_jina (bool): Whether to use Jina for extraction.
        snippet (Optional[str]): The snippet to search for.

    Returns:
        str: Extracted text or context.
    """
    try:
        # print(f"extract_text_from_url use_jina: {use_jina}\n")
        if use_jina:
            jina_headers = {
                'Authorization': f'Bearer {jina_api_key}',
                'X-Return-Format': 'markdown',
                # 'X-With-Links-Summary': 'true'
            }
            response = requests.get(f'https://r.jina.ai/{url}', headers=jina_headers).text
            # Remove URLs
            pattern = r"\(https?:.*?\)|\[https?:.*?\]"
            text = re.sub(pattern, "", response).replace('---','-').replace('===','=').replace('   ',' ').replace('   ',' ')
            print("use jina to extract text successfully")
        else:
            # print(f"don't use jina to extract text")
            response = session.get(url, timeout=20)  # Set timeout to 20 seconds
            response.raise_for_status()  # Raise HTTPError if the request failed
            # Determine the content type
            content_type = response.headers.get('Content-Type', '')
            if 'pdf' in content_type:
                # If it's a PDF file, extract PDF text
                print("Extracting text from PDF...")
                return extract_pdf_text(url)
            # Try using lxml parser, fallback to html.parser if unavailable
            try:
                # print("use lxml parser to extract text")
                soup = BeautifulSoup(response.text, 'lxml')
            except Exception:
                print("lxml parser not found or failed, falling back to html.parser")
                soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text(separator=' ', strip=True)

        if snippet:
            success, context = extract_snippet_with_context(text, snippet)
            if success:
                print("use extract_snippet_with_context to extract text successfully")
                return context
            else:
                print("use extract_snippet_with_context to extract text failed")
                return text
        else:
            # print("no snippet provided")
            # If no snippet is provided, return directly
            return text[:8000]
            # return text[:10000]
    except requests.exceptions.HTTPError as http_err:
        return f"HTTP error occurred: {http_err}"
    except requests.exceptions.ConnectionError:
        return "Error: Connection error occurred"
    except requests.exceptions.Timeout:
        return "Error: Request timed out after 20 seconds"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

def fetch_page_content(urls, max_workers=24, use_jina=False, jina_api_key=None, snippets: Optional[dict] = None):
    """
    Concurrently fetch content from multiple URLs.

    Args:
        urls (list): List of URLs to scrape.
        max_workers (int): Maximum number of concurrent threads.
        use_jina (bool): Whether to use Jina for extraction.
        snippets (Optional[dict]): A dictionary mapping URLs to their respective snippets.

    Returns:
        dict: A dictionary mapping URLs to the extracted content or context.
    """
    results = {}
    max_workers=20
    print(f"max_workers: {max_workers}")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Use tqdm to display a progress bar
        futures = {
            executor.submit(extract_text_from_url, url, use_jina, jina_api_key,snippets.get(url) if snippets else None): url
            for url in urls
        }
        for future in tqdm(concurrent.futures.as_completed(futures), desc="Fetching URLs", total=len(urls)):
            url = futures[future]
            # try:
            #     data = future.result()
            #     results[url] = data
            # except Exception as exc:
            #     results[url] = f"Error fetching {url}: {exc}"
            # time.sleep(0.2)  # Simple rate limiting

            for _ in range(5): # max retry 5 times
                try:
                    data = future.result()
                    results[url] = data
                    break
                except Exception as exc:
                    results[url] = f"Error fetching {url}: {exc}"
                    time.sleep(0.2)
    return results


proxies = {
    "http": "http://127.0.0.1:7890",
    "https": "http://127.0.0.1:7890"
}


def google_web_search(query, subscription_key, endpoint, market='en-US', language='en', exclude_urls=[],timeout=2000):
    """
    Perform a search using the Bing Web Search API with a set timeout.

    Args:
        query (str): Search query.
        subscription_key (str): Subscription key for the Bing Search API.
        endpoint (str): Endpoint for the Bing Search API.
        market (str): Market, e.g., "en-US" or "zh-CN".
        language (str): Language of the results, e.g., "en".
        timeout (int or float or tuple): Request timeout in seconds.
                                         Can be a float representing the total timeout,
                                         or a tuple (connect timeout, read timeout).

    Returns:
        dict: JSON response of the search results. Returns None or raises an exception if the request times out.
    """
    
    if exclude_urls:
        
        for site in exclude_urls:
            query += f" -site:{site}"
        print(f"qeury: {query}, exclude_urls: {exclude_urls}")
    # query = query + " site:en.wikipedia.org"
    # print(f"query: {query}")
    payload = json.dumps({
        "q": query,  # 设置查询内容
        "num": 11,
        "mkt": market,  # 设置市场
        "setLang": language,  # 设置语言
        "textDecorations": True,  # 启用文本装饰
        "textFormat": "HTML"  # 设置文本格式
    })
    print(f"query: {query}")

    headers = {
        'X-API-KEY': subscription_key,
        'Content-Type': 'application/json'
    }
    error_cnt = 0
    while True:
        if error_cnt == 20:
            print(f"qery: {query} has tried {error_cnt} times without success, just skip it.")
            break
        try:
            # 发送POST请求
            # response = requests.request("POST", endpoint, headers=headers, data=payload, proxies=proxies, timeout=timeout)
            response = requests.request("POST", endpoint, headers=headers, data=payload, timeout=timeout)
            response.raise_for_status()  # Raise exception if the request failed 检查响应的状态码。如果返回的状态码是 4xx 或 5xx（表示客户端或服务器错误），它将引发 requests.exceptions.HTTPError 异常
            search_results = response.json() #
            return search_results
        except Timeout:
            error_cnt += 1
            print(f"error_cnt: {error_cnt}, Bing Web Search request timed out ({timeout} seconds) for query: {query}")
            time.sleep(5)
            # return {}  # Or you can choose to raise an exception
        except requests.exceptions.RequestException as e:
            error_cnt += 1
            print(f"error_cnt: {error_cnt}, Error occurred during Bing Web Search request: {e}, payload: {payload}")
            time.sleep(5)
            # return {}


def extract_pdf_text(url):
    """
    Extract text from a PDF.

    Args:
        url (str): URL of the PDF file.

    Returns:
        str: Extracted text content or error message.
    """
    try:
        response = session.get(url, timeout=20)  # Set timeout to 20 seconds
        if response.status_code != 200:
            return f"Error: Unable to retrieve the PDF (status code {response.status_code})"
        
        # Open the PDF file using pdfplumber
        with pdfplumber.open(BytesIO(response.content)) as pdf:
            full_text = ""
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text
        
        # Limit the text length
        cleaned_text = ' '.join(full_text.split()[:600])
        return cleaned_text
    except requests.exceptions.Timeout:
        return "Error: Request timed out after 20 seconds"
    except Exception as e:
        return f"Error: {str(e)}"

def extract_relevant_info(search_results):
    """
    Extract relevant information from Bing search results.

    Args:
        search_results (dict): JSON response from the Bing Web Search API.

    Returns:
        list: A list of dictionaries containing the extracted information.
    """
    useful_info = []

    if search_results == None:
        return useful_info

    if 'organic' in search_results : # value 通常是一个列表，包含了搜索结果的每个页面信息
        for id, result in enumerate(search_results['organic']):
            info = {
                'id': id + 1,  # Increment id for easier subsequent operations 为每个结果分配一个 id，id + 1 是为了让 ID 从 1 开始，而不是从 0 开始。这对后续操作更直观
                'title': result.get('title', ''), # 每个搜索结果中提取标题
                'url': result.get('link', ''), # 每个搜索结果中提取 URL
                'site_name': result.get('siteName', ''), # 每个搜索结果中提取站点名称
                'date': result.get('datePublished', '').split('T')[0], # 提取搜索结果的发布时间
                'snippet': result.get('snippet', ''),  # Remove HTML tags : 提取搜索结果的简短描述（即摘要或片段），result.get('snippet', '')。这里的 snippet 可能包含 HTML 标签，因此需要在后续的处理中可能会清除这些标签
                # Add context content to the information
                'context': ''  # Reserved field to be filled later
            }
            useful_info.append(info)
    else:
        print("No organic results found.")
    print(f"len of useful_info: {len(useful_info)}")
    return useful_info
