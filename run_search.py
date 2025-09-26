import argparse
import json
import time

from search.rerankers import load_reranker
from search.retrievers import load_retriever


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--retriever", type=str, required=True)
    parser.add_argument("--retriever_top_k", type=int, default=10)
    parser.add_argument("--retriever_kwargs", type=str, default="{}")
    parser.add_argument("--reranker", type=str)
    parser.add_argument("--reranker_max_tokens", type=int, default=1024)
    parser.add_argument("--reranker_batch_size", type=int, default=1)
    parser.add_argument("--reranker_kwargs", type=str, default="{}")
    parser.add_argument("--print_top_k", type=int, default=10)
    parser.add_argument("--print_excerpt_length", type=int, default=200)
    args = parser.parse_args()

    print("Loading retriever")
    retriever = load_retriever(args.retriever, default_k=args.retriever_top_k, **json.loads(args.retriever_kwargs))
    print(f"Loaded retriever: {retriever.__class__.__name__}")

    if args.reranker is not None:
        print("Loading reranker")
        reranker = load_reranker(
            args.reranker,
            max_length=args.reranker_max_tokens,
            batch_size=args.reranker_batch_size,
            **json.loads(args.reranker_kwargs),
        )
        print(f"Loaded reranker: {reranker.__class__.__name__}")
    else:
        reranker = None
        print("No reranker loaded")

    print(f"Calling retriever for: {args.query}")
    retriever_start_time = time.time()
    documents = retriever(args.query)
    retriever_elapsed_time = time.time() - retriever_start_time
    print(f"Retriever took {retriever_elapsed_time:.3f} seconds to retrieve {len(documents)} documents")

    if reranker is not None:
        print(f"Calling reranker for {len(documents)} documents")
        reranker_start_time = time.time()
        documents, scores = reranker(args.query, documents)
        reranker_elapsed_time = time.time() - reranker_start_time
        print(f"Reranker took {reranker_elapsed_time:.3f} seconds to rerank {len(documents)} documents")
    else:
        scores = None

    for i, document in enumerate(documents[: args.print_top_k]):
        excerpt = document.text[: args.print_excerpt_length].replace("\n", " ")

        if scores is not None:
            print(f"Rank {i + 1} (Reranker Score: {scores[i]:.4f}):")
        else:
            print(f"Rank {i + 1}:")

        print(f"  URL: {document.url}")
        print(f"  Excerpt: {excerpt}...")
        print("")
