import hashlib
import json
import logging
import os
import re
import time
from typing import Iterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from httpx import BasicAuth, Client
from logtail import LogtailHandler
from openai import OpenAI
from openai.types import Completion
from pydantic import BaseModel
from transformers import PreTrainedTokenizer

from search.data import Document
from search.rerankers import load_reranker
from search.retrievers import load_retriever
from utils import extract_between_tags, load_tokenizer
from utils.constants import BEGIN_SEARCH_QUERY, BEGIN_SEARCH_RESULT, END_SEARCH_QUERY, END_SEARCH_RESULT
from utils.prompts import get_qa_instruction
from utils.summarizer import Summarizer


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

logtail_host = os.getenv("LOGTAIL_HOST")
if logtail_host:
    logtail_handler = LogtailHandler(source_token=os.getenv("LOGTAIL_TOKEN"), host=logtail_host)
    logger.addHandler(logtail_handler)
    logger.info("Logtail handler added (host: %s)", logtail_host)


class EvaluateRequest(BaseModel):
    query: str
    iid: str


class EvaluateResponse(BaseModel):
    query_id: str
    generated_response: str


class RunRequest(BaseModel):
    question: str


app = FastAPI()

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)


def run_generation_openai(
    prompt: str,
    client: OpenAI,
    model: str,
    tokenizer: PreTrainedTokenizer,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k_sampling: int,
    stop: list[str],
) -> Completion:
    output = client.completions.create(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        stop=[END_SEARCH_QUERY, tokenizer.eos_token],
        temperature=temperature,
        top_p=top_p,
        extra_body={"top_k": top_k_sampling, "include_stop_str_in_output": True},
    )
    return output


class OpenAISummarizer(Summarizer):
    def __init__(
        self,
        client: OpenAI,
        model: str,
        tokenizer: PreTrainedTokenizer,
        top_k: int,
        max_tokens_per_webpage: int,
        max_tokens: int = 8192,
        temperature: float = 0.6,
        top_p: float = 0.95,
    ):
        super().__init__(
            llm=None,
            tokenizer=tokenizer,
            top_k=top_k,
            max_tokens_per_webpage=max_tokens_per_webpage,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        self.client = client
        self.model = model

    @staticmethod
    def _parse_result(output: str) -> str:
        split_str = "**Final Information**"
        if split_str in output:
            extracted_text = output.split(split_str)[-1].replace("\n", "").strip("```").strip()
        else:
            extracted_text = None

        return extracted_text

    def __call__(
        self,
        previous_reasoning: str | None,
        search_query: str,
        documents: dict[str, dict[str, str]],
        max_retry: int = 10,
    ) -> str:
        if previous_reasoning is None:
            prompt = self._generate_initial_search_summary_prompt(search_query, documents)
        else:
            prompt = self._generate_summary_prompt(previous_reasoning, search_query, documents)

        messages = [{"role": "user", "content": prompt}]

        raw_output = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            stop=None,
            temperature=self.temperature,
            top_p=self.top_p,
        )

        result = self._delete_invalid_spaces(self._parse_result(raw_output.choices[0].message.content))

        for _ in range(max_retry):
            valid_ids = list(documents.keys())
            validation = self._validate_citation_format(result, valid_ids)
            if validation["is_valid"]:
                break

            retry_raw_output = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                stop=None,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            result = self._delete_invalid_spaces(self._parse_result(retry_raw_output.choices[0].message.content))

        return result


try:
    model_path = os.getenv("MODEL_PATH")
    summarizer_model_path = os.getenv("SUMMARIZER_MODEL_PATH")
    reranker_model_path = os.getenv("RERANKER_MODEL_PATH")
    logger.info("MODEL_PATH: %s", model_path)
    logger.info("SUMMARIZER_MODEL_PATH: %s", summarizer_model_path)
    logger.info("RERANKER_MODEL_PATH: %s", reranker_model_path)

    max_search_limit = 5
    max_turns = 15
    max_rollouts = 3
    max_tokens_per_webpage = 4096
    max_tokens = 20480
    temperature = 0.6
    top_p = 0.95
    top_k_sampling = 20

    retriever_name = "clueweb22-a"
    retriever_top_k = 300
    retriever_kwargs = "{}"
    max_search_retries = 5

    reranker_name = "qwen3"
    reranker_max_tokens = 4096
    reranker_batch_size = 1
    reranker_kwargs = "{}"

    summarizer_top_k = 10
    summarizer_max_tokens = 8192
    summarizer_temperature = 0.6
    summarizer_top_p = 0.95

    continual_search_queries = set(
        ["...", "query", "Enter your query here", "and", "[query]", "query here", "[Your search query]"]
    )

    logger.info("Setting up OpenAI API...")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_api_base = os.getenv("OPENAI_API_BASE")
    logger.info("OPENAI_API_BASE: %s", openai_api_base)
    http_client = None
    if os.getenv("OPENAI_API_USERNAME"):
        logger.info("Configuring custom authentication for OpenAI API...")
        auth = BasicAuth(username=os.getenv("OPENAI_API_USERNAME"), password=os.getenv("OPENAI_API_PASSWORD"))
        http_client = Client(auth=auth)

    client = OpenAI(api_key=openai_api_key, base_url=openai_api_base, http_client=http_client)

    logger.info("Setting up OpenAI API for summarizer...")
    summarizer_openai_base = os.getenv("SUMMARIZER_OPENAI_API_BASE")
    summarizer_openai_key = os.getenv("SUMMARIZER_OPENAI_API_KEY")
    logger.info("SUMMARIZER_OPENAI_API_BASE: %s", summarizer_openai_base)
    summarizer_http_client = None
    if os.getenv("SUMMARIZER_OPENAI_API_USERNAME"):
        logger.info("Configuring custom authentication for Summarizer OpenAI API...")
        summarizer_auth = BasicAuth(
            username=os.getenv("SUMMARIZER_OPENAI_API_USERNAME"), password=os.getenv("SUMMARIZER_OPENAI_API_PASSWORD")
        )
        summarizer_http_client = Client(auth=summarizer_auth)

    summarizer_client = OpenAI(
        api_key=summarizer_openai_key, base_url=summarizer_openai_base, http_client=summarizer_http_client
    )

    logger.info("Loading tokenizer...")
    tokenizer = load_tokenizer(model_path)
    summarizer_tokenizer = load_tokenizer(summarizer_model_path)

    logger.info("Loading retriever...")
    retriever = load_retriever(retriever_name, default_k=retriever_top_k, **json.loads(retriever_kwargs))

    reranker = None
    if reranker_name is not None:
        logger.info("Loading reranker...")
        reranker = load_reranker(
            reranker_name,
            max_length=reranker_max_tokens,
            batch_size=reranker_batch_size,
            **json.loads(reranker_kwargs),
        )

    logger.info("Loading summarizer...")
    summarizer = OpenAISummarizer(
        client=summarizer_client,
        model=summarizer_model_path,
        tokenizer=summarizer_tokenizer,
        top_k=summarizer_top_k,
        max_tokens_per_webpage=max_tokens_per_webpage,
        max_tokens=summarizer_max_tokens,
        temperature=summarizer_temperature,
        top_p=summarizer_top_p,
    )
    logger.info("Setup completed.")
except Exception as e:
    logger.error("Error during setup: %s", e)
    raise e


def make_intermediate_steps(output: str) -> str:
    output = re.sub(r"</?think>", "", output)
    output = re.sub(r"\n\n+", "\n", output)
    return "|||---|||".join(output.split("\n"))


def extract_final_answer(output: str) -> str:
    marker = "**Final Information**"
    if marker in output:
        return output.split(marker)[-1].strip()
    else:
        return ""


def generate_ref_id(existing_ids: set, reranked_webpages: list[Document]) -> dict[str, dict[str, str]]:
    result = {}

    for webpage in reranked_webpages:
        hash_object = hashlib.md5(webpage.text.encode()).hexdigest()
        for i in range(len(hash_object) - 3):
            ref_id = "#" + hash_object[i : i + 4]
            if ref_id not in existing_ids and ref_id not in result.keys():
                result[ref_id] = {"text": webpage.text, "url": webpage.url}
                break

    return result


def extract_citations(text: str, executed_search_urls: dict[str, str]) -> tuple[dict[str, int], list[str]]:
    ref_id2idx = {}
    urls = []

    for match in re.finditer(r"\#[a-f0-9]{4}", text):
        ref_id = match.group(0)
        if ref_id in executed_search_urls and ref_id not in ref_id2idx:
            ref_id2idx[ref_id] = len(ref_id2idx)
            urls.append(executed_search_urls[ref_id])

    return ref_id2idx, urls


def run_inference(question: str) -> Iterator[dict[str, str | bool | None]]:
    logger.info("Processing question: %s", question)

    logger.info("Performing initial search...")
    for search_retry_count in range(max_search_retries):
        try:
            search_results = retriever(question)
            logger.info("Retrieved %d documents.", len(search_results))
            break
        except Exception:
            if search_retry_count < max_search_retries - 1:
                interval = 10 * 2**search_retry_count
                logger.info("Search failed. Trying after %d seconds...", interval)
                time.sleep(interval)
    else:
        search_results = []
        logger.warning("Retriever failed. Proceeding with zero documents.")

    if reranker is not None and len(search_results) > 0:
        logger.info("Reranking search results...")
        reranked_results, _ = reranker(question, search_results)
        logger.info("Reranked %d documents.", len(reranked_results))
    else:
        reranked_results = search_results

    for rollout_count in range(max_rollouts):
        logger.info("Starting rollout %d...", rollout_count + 1)

        logger.info("Generating initial search summary...")
        initial_search_documents = generate_ref_id(set(), reranked_results)
        initial_search_summary = summarizer(
            previous_reasoning=None, search_query=question, documents=initial_search_documents
        )
        logger.info("Initial search summary generated.")

        instruction = get_qa_instruction(max_search_limit, question, initial_search_summary)
        prompt = [{"role": "user", "content": instruction}]
        prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)

        output = "" if rollout_count == 0 else "Restarting...\n"
        search_count = 0
        executed_search_queries = set()
        executed_search_urls = {ref_id: data["url"] for ref_id, data in initial_search_documents.items()}

        for turn in range(max_turns):
            logger.info("Starting turn %d...", turn + 1)

            logger.info("Generating response...")
            logger.info("Prompt: %s", prompt + output)
            turn_output = (
                run_generation_openai(
                    prompt=prompt + output,
                    client=client,
                    model=model_path,
                    tokenizer=tokenizer,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k_sampling=top_k_sampling,
                    stop=[END_SEARCH_QUERY, tokenizer.eos_token],
                )
                .choices[0]
                .text
            )
            logger.info("Response generated.")
            logger.info("Output: %s", turn_output)

            output += turn_output

            search_query = extract_between_tags(turn_output, BEGIN_SEARCH_QUERY, END_SEARCH_QUERY)

            if search_query and output.rstrip().endswith(END_SEARCH_QUERY):
                if search_count < max_search_limit and search_query not in executed_search_queries:
                    logger.info("Performing search %d with query: %s", search_count + 1, search_query)

                    if search_query in continual_search_queries:
                        logger.info('Detected continual search query: "%s"', search_query)
                    else:
                        for search_retry_count in range(max_search_retries):
                            try:
                                search_results = retriever(search_query)
                                logger.info("Retrieved %d documents.", len(search_results))
                                break
                            except Exception:
                                if search_retry_count < max_search_retries - 1:
                                    interval = 10 * 2**search_retry_count
                                    logger.info("Search failed. Trying after %d seconds...", interval)
                                    time.sleep(interval)
                        else:
                            search_results = []
                            logger.warning("Retriever failed. Proceeding with zero documents.")

                        if reranker is not None and len(search_results) > 0:
                            logger.info("Reranking search results...")
                            reranked_results, _ = reranker(search_query, search_results)
                            logger.info("Reranked %d documents.", len(reranked_results))
                        else:
                            reranked_results = search_results

                        existing_ids = executed_search_urls.keys()
                        search_documents = generate_ref_id(existing_ids, reranked_results)
                        executed_search_urls.update({ref_id: data["url"] for ref_id, data in search_documents.items()})

                        search_count += 1
                        executed_search_queries.add(search_query)

                        all_reasoning_steps = output.replace("\n\n", "\n").split("\n")
                        all_reasoning_steps = [f"Step {i + 1}: {step}" for i, step in enumerate(all_reasoning_steps)]

                        if len(all_reasoning_steps) < 5:
                            reasoning_steps = all_reasoning_steps
                        else:
                            reasoning_steps = []
                            for i, reasoning_step in enumerate(all_reasoning_steps):
                                if (
                                    i == 0
                                    or i > len(all_reasoning_steps) - 4
                                    or BEGIN_SEARCH_QUERY in reasoning_step
                                    or BEGIN_SEARCH_RESULT in reasoning_step
                                ):
                                    reasoning_steps.append(reasoning_step)
                                elif reasoning_steps[-1] != "...":
                                    reasoning_steps.append("...")

                        logger.info("Generating search summary...")
                        webpage_summary = summarizer(
                            previous_reasoning="\n\n".join(reasoning_steps),
                            search_query=search_query,
                            documents=search_documents,
                            max_retry=20,
                        )
                        logger.info("Search summary generated.")
                        logger.info("Summarizer output: %s", webpage_summary)

                        append_text = f"\n\n{BEGIN_SEARCH_RESULT}{webpage_summary}{END_SEARCH_RESULT}\n\n"
                        output += append_text
                elif search_count >= max_search_limit:
                    limit_message = f"\n{BEGIN_SEARCH_RESULT}\nThe maximum search limit is exceeded. You are not allowed to search.\n{END_SEARCH_RESULT}\n"
                    output += limit_message
                elif search_query in executed_search_queries:
                    limit_message = f"\n{BEGIN_SEARCH_RESULT}\nYou have searched this query. Please refer to previous results.\n{END_SEARCH_RESULT}\n"
                    output += limit_message

                intermediate_steps = make_intermediate_steps(output)
                ref_id2idx, urls = extract_citations(intermediate_steps, executed_search_urls)
                for ref_id, idx in ref_id2idx.items():
                    intermediate_steps = intermediate_steps.replace(ref_id, f"[{idx + 1}]")

                yield {
                    "intermediate_steps": intermediate_steps,
                    "final_report": None,
                    "is_intermediate": True,
                    "complete": False,
                    "citations": urls,
                }
            else:
                intermediate_steps = make_intermediate_steps(output)
                ref_id2idx, urls = extract_citations(intermediate_steps, executed_search_urls)
                for ref_id, idx in ref_id2idx.items():
                    intermediate_steps = intermediate_steps.replace(ref_id, f"[{idx + 1}]")

                yield {
                    "intermediate_steps": intermediate_steps,
                    "final_report": None,
                    "is_intermediate": False,
                    "complete": False,
                    "citations": urls,
                }
                break

        final_report = extract_final_answer(output)
        if final_report != "":
            logger.info("Final answer found. Exiting rollouts.")
            break
        elif rollout_count < max_rollouts - 1:
            logger.info("No final answer found. Starting a new rollout...")
    else:
        logger.warning("No final answer found after maximum rollouts. Returning an empty answer.")

    intermediate_steps = make_intermediate_steps(output)
    ref_id2idx, urls = extract_citations(intermediate_steps + final_report, executed_search_urls)
    for ref_id, idx in ref_id2idx.items():
        intermediate_steps = intermediate_steps.replace(ref_id, f"[{idx + 1}]")
        final_report = final_report.replace(ref_id, f"[{idx + 1}]")

    logger.info("Cleaning up reference IDs...")
    intermediate_steps = re.sub(r"\#[a-zA-Z0-9_]{4,}", "", intermediate_steps)
    final_report = re.sub(r"\#[a-zA-Z0-9_]{4,}", "", final_report)
    for match in re.finditer(r"\((\s*\[\d+\]\s*,?)*\)", intermediate_steps + final_report):
        match_text = match.group(0)
        formatted_text = match_text.replace(",", "").replace(" ", "").strip("()")
        intermediate_steps = intermediate_steps.replace(match_text, formatted_text)
        final_report = final_report.replace(match_text, formatted_text)

    logger.info("Final report: %s", final_report)

    yield {
        "intermediate_steps": intermediate_steps,
        "final_report": final_report,
        "is_intermediate": False,
        "complete": True,
        "citations": urls,
    }


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/evaluate")
def evaluate(request: EvaluateRequest) -> EvaluateResponse:
    generated_response = ""
    try:
        for item in run_inference(request.query):
            if item["complete"]:
                generated_response = item["final_report"]
                break
    except Exception as e:
        logger.error("Error during evaluation: %s", e)

    return EvaluateResponse(query_id=request.iid, generated_response=generated_response)


def response_streamer(question: str) -> Iterator[str]:
    item = {"intermediate_steps": None, "final_report": None, "is_intermediate": True, "complete": False}
    try:
        for item in run_inference(question):
            yield f"data: {json.dumps(item)}\n"
    except Exception as e:
        logger.error("Error during evaluation: %s", e)
        item["error"] = str(e)
        item["complete"] = True
        yield f"data: {json.dumps(item)}\n"


@app.post("/run")
def run(request: RunRequest) -> StreamingResponse:
    return StreamingResponse(response_streamer(request.question), media_type="text/event-stream")
