def get_qa_instruction(MAX_SEARCH_LIMIT: int) -> str:
    return (
        "You are a reasoning assistant with the ability to perform web searches to help "
        "you answer the user's question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n\n"
        f"Whenever you encounter a topic, fact, or piece of information you are uncertain about or need further details on, please perform a search to gather more accurate, up-to-date, or specific information. You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- When done searching, continue your reasoning.\n"
        "- Do not generate <|begin_search_result|> and <|end_search_result|> tags yourself.\n\n"
    )


def get_task_instruction(question: str) -> str:
    user_prompt = (
        "Please answer the following question. You should think step by step to solve it.\n\n"
        "Your answer should be clear, detailed, and insightful."
        "Your answer should be written in 50-100 words and structured in 2-5 sentences. "
        "Do NOT respond with a single word, phrase or sentence.\n"
        "Provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n"
        f"Question:\n{question}\n\n"
    )
    return user_prompt
