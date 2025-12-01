# An Open and Reproducible Deep Research Agent for Long-Form Question Answering

This repository hosts a deep research system submitted to the MMU-RAG competition at NeurIPS 2025.
The system is capable of generating report-style, long-form answers to arbitrary questions.

### Generate Answer Rollouts

Generate multiple answer candidates for each question using the base model (`Qwen/Qwen3-Next-80B-A3B-Thinking-FP8`). This step creates 20 different responses per question.

```bash
export RETRIEVER_API_KEY=<CLUE_WEB_API_KEY>
export VLLM_FLASH_ATTN_VERSION=2

python generate_rollouts.py \
    --data_path <DATA_PATH> \
    --output_dir_base  <YOUR_OUTPUT_PATH> \
    --model_path Qwen/Qwen3-Next-80B-A3B-Thinking-FP8 \
    --retriever clueweb22-a \
    --reranker qwen3 \
    --max_search_limit 5 \
    --max_tokens_per_webpage 4096 \
    --top_k_sampling 20 \
    --rollout_num 20 \
    --retriever_top_k 100 \
    --reranker_max_tokens 4096 \
    --auto_resume
```

### Evaluate Generated Answers

Evaluate the generated answers using an LLM-as-a-judge approach. This assigns preference scores based on clarity, insightfulness, and factuality metrics.

Before running the evaluation, configure your Azure OpenAI credentials in a `.env` file:
```
AZURE_DEPLOYMENT="o3-mini"
AZURE_OPENAI_ENDPOINT=""
AZURE_OPENAI_API_KEY=""
AZURE_OPENAI_API_VERSION=""
```

Then run the evaluation. Note that `--root_path` should be the same as `--output_dir_base` you specified in the previous command. By setting `--filtering_criteria finished valid_citation_format`, unfinished rollouts and rollouts containing citation format errors will be filtered out:
```bash
python evaluate_rollouts.py \
    --root_path <ROOT_PATH> \
    --eval_kpr \
    --max_key_points 10 \
    --eval_clarity_and_insightfulness \
    --output_path <YOUR_OUTPUT_PATH> \
    --filtering_criteria finished valid_citation_format
```

### Construct Preference Pairs

Construct preference pairs by selecting the best and worst answers for each question based on their preference scores. The script ensures a minimum score gap (`--min_score_gap`) between chosen and rejected answers.
```bash
python format_preference_data.py \
    --evaluated_rollouts_file  \
    --output_path  \
    --min_score_gap 0.3 \
    --instruction_search_limit 5 \
    --save_jsonl
```

## Training

## Inference

You can run an inference server with the trained model, using a Docker container.

We assume that the LLMs for generating reasoning steps and summarizing retrieved web page contents are running on remote vLLM servers.
Please set the URLs of those vLLM endpoints to the `OPENAI_API_BASE` and `SUMMARIZER_OPENAI_API_BASE` build arguments.

```bash
docker build --platform linux/amd64 -t efficient-deep-research \
--build-arg MODEL_PATH=efficient-deep-research/gap_0.3_beta_0.5_lora_ckpt_56_merged_FP8 \
--build-arg SUMMARIZER_MODEL_PATH=Qwen/Qwen3-Next-80B-A3B-Thinking-FP8 \
--build-arg OPENAI_API_BASE=http://xxx.xxx.xxx.xxx:8000/v1 \
--build-arg SUMMARIZER_OPENAI_API_BASE=http://xxx.xxx.xxx.xxx:8000/v1 \
--build-arg RETRIEVER_API_KEY=<CLUE_WEB_API_KEY>
.
```

Then run a container:

```bash
docker run --gpus 1 --rm -p 5027:5027 efficient-deep-research
```

Send a request to the API running in the container:

```bash
# Get a static (non-streaming) response
curl http://127.0.0.1:5027/evaluate -H "Content-Type: application/json" -d '{"query": "Explain gravity", "iid": "123"}'

# Get a streaming response
curl http://127.0.0.1:5027/run -H "Content-Type: application/json" -d '{"question": "Explain why the sky is blue"}'
```


## License

Apache License 2.0
