# An Open and Reproducible Deep Research Agent for Long-Form Question Answering

This repository hosts a deep research system submitted to the MMU-RAG competition at NeurIPS 2025.
The system is capable of generating report-style, long-form answers to arbitrary questions.

### Generate Answer Rollouts

Generate multiple answer candidates for each question using the base model (`Qwen/Qwen3-Next-80B-A3B-Thinking-FP8`). This step creates 20 different responses per question through sampling.

```bash
export RETRIEVER_API_KEY=<CLUE_WEB_API_KEY>
export VLLM_FLASH_ATTN_VERSION=2

python generate_rollouts.py \
    --data_path  \
    --output_dir_base  \
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

Then run the evaluation:
```bash
python evaluate_rollouts.py \
    --root_path  \
    --eval_kpr \
    --max_key_points 10 \
    --eval_clarity_and_insightfulness \
    --output_path  \
    --filtering_criteria finished
```

### Construct Preference Pairs

Construct preference pairs by selecting the best and worst answers for each question based on their preference scores. The script filters out malformed responses and ensures a minimum score gap (`--min_score_gap`) between chosen and rejected answers.
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

## License
