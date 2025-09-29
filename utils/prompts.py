def get_qa_instruction(MAX_SEARCH_LIMIT: int) -> str:
    return (
        "You are a reasoning assistant with the ability to perform web searches to help "
        "you answer the user's question accurately.\n\n"
        "To give you a starting point, an initial web search has already been performed based on the user's question. "
        "Please analyze this initial information and use it as the foundation for your reasoning. "
        "If it is insufficient or you need more details, use your own search tool.\n\n"
        "You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n\n"
        f"Whenever you encounter a topic, fact, or piece of information you are uncertain about or need further details on, please perform a search to gather more accurate, up-to-date, or specific information. You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n\n"
        "Once you have all the information you need, continue your reasoning to answer the user's question.\n\n"
        "Remember:\n"
        "- First, analyze the initial search result provided at the start.\n"
        "- The initial search does not count towards your search limit.\n"
        "- Use <|begin_search_query|> to request an additional web search  and end with <|end_search_query|>.\n"
        "- Once you formulate a search query, you must execute it immediately with <|begin_search_query|> and <|end_search_query|>.\n"
        "- When done searching, continue your reasoning.\n"
        "- Do not generate <|begin_search_result|> and <|end_search_result|> tags yourself.\n"
    )


def get_task_instruction(question: str, initial_search_result: str) -> str:
    user_prompt = (
        "Based on the initial information provided below, please answer the question. You should think step by step to solve it.\n\n"
        "Your answer should be clear, detailed, and insightful."
        "Your answer should be written in 50-100 words and structured in 2-5 sentences. "
        "Do NOT respond with a single word, phrase or sentence.\n"
        "Provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n"
        f"Question:\n{question}\n"
        f"Initial Web Search Result:\n{initial_search_result}\n\n"
    )
    return user_prompt
