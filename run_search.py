import argparse
import json

from search.rerankers import ContextualAIReranker
from search.retrievers import FinewWebRetriever


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--reranker_kwargs", type=str, default="{}")
    args = parser.parse_args()

    retriever = FinewWebRetriever(default_k=args.k)
    reranker = ContextualAIReranker()

    documents = retriever(args.query, args.k)
    reranked_documents, scores = reranker(args.query, documents, **json.loads(args.reranker_kwargs))

    for i, document in enumerate(reranked_documents):
        print(f"Rank {i + 1} (Score: {scores[i]:.4f}):")
        print(f"  URL: {document.url}")
        print(f"  Excerpt: {document.text[:200]}...")
        print("")
