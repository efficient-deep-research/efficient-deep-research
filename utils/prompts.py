def get_qa_instruction(MAX_SEARCH_LIMIT: int) -> str:
    return (
        "You are a reasoning assistant with the ability to perform web searches to help you answer the user's question accurately.\n\n"
        
        "**Initial Search Result:**\n"
        "To give you a starting point, an initial web search has already been performed based on the user's question. "
        "Please analyze this initial information first. "
        "However, note that it may be insufficient or incomplete, so you are encouraged to perform additional searches to gather more comprehensive information.\n\n"

        "**Available Tools:**\n"
        "You have access to a web search tool:\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>\n"
        "- The system will then search and analyze relevant web pages, and provide you with helpful information in the format: <|begin_search_result|> ...search results... <|end_search_result|>\n"
        "- Do NOT generate <|begin_search_result|> and <|end_search_result|> tags yourself\n\n"

        "**Search Guidelines:**\n"
        f"- You can perform up to {MAX_SEARCH_LIMIT} additional searches (the initial search does not count towards this limit)\n"
        "- Whenever you encounter a topic, fact, or piece of information you are uncertain about or need further details on, perform a search to gather more accurate, up-to-date, or specific information\n"
        "- You can repeat the search process multiple times if necessary\n"
        "- Once you formulate a search query, you must execute it immediately with <|begin_search_query|> and <|end_search_query|>\n\n"
        
        "**Citation Requirements:**\n"
        "When you use information from the search results:\n"
        "- Each sentence in the search results ends with a webpage identifier in the format (#WEBPAGE_ID)\n"
        "- You MUST include the (#WEBPAGE_ID) citation at the end of any statement that uses information from that source\n"
        "- Place the citation immediately after the relevant information, before the period or other punctuation\n"
        "- Example: If the search result says 'Women earned 80.5 cents for every $1 earned by men in 2016 (#6702).', "
        "you should write: 'According to the data, women earned 80.5 cents for every dollar earned by men in 2016 (#6702).'\n"
        "- If you combine information from multiple sources in one statement, include all relevant citations: 'This phenomenon is observed across multiple studies (#6702)(#814c).'\n"
        "- Always preserve the exact WEBPAGE_ID from the source when citing\n\n"
    )


def get_task_instruction(question: str, initial_search_result: str) -> str:
    user_prompt = (
        "Based on the initial information provided below, please answer the question. You should think step by step to solve it.\n\n"
        "Your answer should be clear, detailed, and insightful."
        "Your answer should be written in 50-100 words and structured in 2-5 sentences. "
        "Do NOT answer with a single word, phrase or sentence.\n"
        "Provide your answer in the format \\boxed{YOUR_ANSWER}.\n\n"
        f"Question:\n{question}\n"
        f"Initial Web Search Result:\n{initial_search_result}\n\n"
    )
    return user_prompt
