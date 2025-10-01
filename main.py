import json
import re
from typing import Iterator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from search.rerankers import load_reranker
from search.retrievers import load_retriever
from utils import extract_between_tags, load_tokenizer, load_vllm_model, run_generation
from utils.constants import BEGIN_SEARCH_QUERY, BEGIN_SEARCH_RESULT, END_SEARCH_QUERY, END_SEARCH_RESULT
from utils.prompts import get_qa_instruction, get_task_instruction
from utils.summarizer import Summarizer


class EvaluateRequest(BaseModel):
    query: str
    iid: str


class EvaluateResponse(BaseModel):
    query_id: str
    generated_response: str


class RunRequest(BaseModel):
    query: str


app = FastAPI()


model_path = "RUC-AIBOX/Qwen-7B-SimpleDeepSearcher"
gpu_memory_utilization = 0.75

max_search_limit = 10
max_turns = 15
max_tokens = 20480
temperature = 0.6
top_p = 0.95
top_k_sampling = 40

retriever_name = "clueweb22-a"
retriever_top_k = 1000
retriever_kwargs = "{}"

reranker_name = "contextualai"
reranker_max_tokens = 1024
reranker_batch_size = 1
reranker_kwargs = "{}"

summarizer_top_k = 10
summarizer_max_tokens = 8192
summarizer_temperature = 0.6
summarizer_top_p = 0.95


print("Loading LLM...")
llm = load_vllm_model(model_path, gpu_memory_utilization=gpu_memory_utilization)
print("Loading tokenizer...")
tokenizer = load_tokenizer(model_path)

print("Loading retriever...")
retriever = load_retriever(retriever_name, default_k=retriever_top_k, **json.loads(retriever_kwargs))

reranker = None
if reranker_name is not None:
    print("Loading reranker...")
    reranker = load_reranker(
        reranker_name, max_length=reranker_max_tokens, batch_size=reranker_batch_size, **json.loads(reranker_kwargs)
    )

print("Loading summarizer...")
summarizer = Summarizer(
    llm=llm,
    top_k=summarizer_top_k,
    max_tokens=summarizer_max_tokens,
    temperature=summarizer_temperature,
    top_p=summarizer_top_p,
)
print("Setup completed.")


def extract_final_answer(output: str) -> str:
    pattern = r"\\boxed\{\\text{(.*?)\}\}"
    match = re.search(pattern, output, flags=re.DOTALL)
    if match:
        return match.group(1).strip()

    pattern_unnested = r"\\boxed\{(.*?)\}"
    match_unnested = re.findall(pattern_unnested, output, flags=re.DOTALL)
    if match_unnested:
        return match_unnested[-1].strip()

    return ""


def run_inference(question: str) -> Iterator[dict[str, str | bool | None]]:
    instruction = get_qa_instruction(max_search_limit)
    user_prompt = get_task_instruction(question)
    prompt = [{"role": "user", "content": instruction + user_prompt}]
    prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)

    output = ""
    search_count = 0
    executed_search_queries = set()

    for _ in range(max_turns):
        turn_output = (
            run_generation(
                prompts=[prompt + output],
                llm=llm,
                tokenizer=tokenizer,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k_sampling=top_k_sampling,
                stop=[END_SEARCH_QUERY, tokenizer.eos_token],
            )[0]
            .outputs[0]
            .text
        )

        output += turn_output

        search_query = extract_between_tags(turn_output, BEGIN_SEARCH_QUERY, END_SEARCH_QUERY)

        if search_query and output.rstrip().endswith(END_SEARCH_QUERY):
            if search_count < max_search_limit and search_query not in executed_search_queries:
                try:
                    search_results = retriever(search_query)
                except Exception:
                    search_results = []

                if reranker is not None and len(search_results) > 0:
                    reranked_results, _ = reranker(search_query, search_results)
                else:
                    reranked_results = search_results

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

                webpage_summary = summarizer(
                    previous_reasonings=["\n\n".join(reasoning_steps)],
                    search_queries=[search_query],
                    documents=[reranked_results],
                    # batch_output_records=batch_output_records,  # Pass the collection list
                )[0]

                append_text = f"\n\n{BEGIN_SEARCH_RESULT}{webpage_summary}{END_SEARCH_RESULT}\n\n"
                output += append_text

            elif search_count >= max_search_limit:
                limit_message = f"\n{BEGIN_SEARCH_RESULT}\nThe maximum search limit is exceeded. You are not allowed to search.\n{END_SEARCH_RESULT}\n"
                output += limit_message
            elif search_query in executed_search_queries:
                limit_message = f"\n{BEGIN_SEARCH_RESULT}\nYou have searched this query. Please refer to previous results.\n{END_SEARCH_RESULT}\n"
                output += limit_message

            yield {
                "intermediate_steps": "|||---|||".join(output.replace("\n\n", "\n").split("\n")),
                "final_report": None,
                "is_intermediate": True,
                "complete": False,
            }
        else:
            yield {
                "intermediate_steps": "|||---|||".join(output.replace("\n\n", "\n").split("\n")),
                "final_report": None,
                "is_intermediate": False,
                "complete": False,
            }
            break

    final_report = extract_final_answer(output)

    yield {
        "intermediate_steps": "|||---|||".join(output.replace("\n\n", "\n").split("\n")),
        "final_report": final_report,
        "is_intermediate": False,
        "complete": True,
    }


@app.post("/evaluate")
def evaluate(request: EvaluateRequest) -> EvaluateResponse:
    generated_response = ""
    for item in run_inference(request.query):
        if item["complete"]:
            generated_response = item["final_report"]
            break

    return EvaluateResponse(query_id=request.iid, generated_response=generated_response)


def response_streamer(question: str) -> Iterator[str]:
    for item in run_inference(question):
        yield f"data: {json.dumps(item)}\n"


@app.post("/run")
def run(request: RunRequest) -> StreamingResponse:
    return StreamingResponse(response_streamer(request.query), media_type="text/event-stream")
