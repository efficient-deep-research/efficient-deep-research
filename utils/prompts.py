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
        "Provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n"
        f"Question:\n{question}\n\n"
    )
    return user_prompt


def get_webpage_to_reasonchain_instruction(
    prev_reasoning: str, search_query: str, document: str
) -> str:
    return f"""**Task Instruction:**

You are tasked with reading and analyzing web pages based on the following inputs: **Previous Reasoning Steps**, **Current Search Query**, and **Searched Web Pages**. Your objective is to extract relevant and helpful information for **Current Search Query** from the **Searched Web Pages** and seamlessly integrate this information into the **Previous Reasoning Steps** to continue reasoning for the original question.

**Guidelines:**

1. **Analyze the Searched Web Pages:**
- Carefully review the content of each searched web page.
- Identify factual information that is relevant to the **Current Search Query** and can aid in the reasoning process for the original question.

2. **Extract Relevant Information:**
- Select the information from the Searched Web Pages that directly contributes to advancing the **Previous Reasoning Steps**.
- Ensure that the extracted information is accurate and relevant.

3. **Output Format:**
- Present the helpful information for current search query: beginning with `**Final Information**` as shown below.
**Final Information**

[Helpful information]

**Inputs:**
- **Previous Reasoning Steps:**  
{prev_reasoning}

- **Current Search Query:**  
{search_query}

- **Searched Web Pages:**  
{document}

Now you should analyze each web page and find helpful information based on the current search query "{search_query}" and previous reasoning steps.
"""


def create_kpr_prompt(key_point, answer):

    return f"""You are given a **single key point** and a **report**.

    Your job is to determine whether the report:
    - **Supports** the key point (it affirms, explains, or reinforces the point),
    - **Omits** the key point (it does not mention or cover this point at all), or
    - **Contradicts** the key point (it says something that disagrees with or negates the point).

    Carefully read the key point and the report.

    Return your answer as a **JSON object** with two fields:
    - "label": One of "Supported", "Omitted", or "Contradicted".
    - "justification": Brief explanation on why you assigned this label.

    Respond strictly in JSON format:
    {{"label": label, "justification": justification}}
    Do **not** add any extra commentary or text outside the JSON.

    ---

    Key Point: {key_point}
    Report: {answer}
    """


def create_eval_criteria_prompt(eval_criteria, question, answer):
    criteria_to_description = {
        "Clarity": "Assess how clearly, rigorously, and analytically distinct the answer is. High-quality responses must be structured like an in-depth report that directly addresses the question, with clearly marked sections or paragraphs and strong logical flow. Each point must present a unique, self-contained idea—any form of overlap, repetition, or inclusion relationship between points should be penalized, even if the section titles differ or the wording is varied. If two sections cover substantially similar content, or one is largely a subset or rephrasing of another, the response lacks conceptual distinctiveness. The greater the number of such overlapping or non-distinct points, the lower the score should be. Superficial variety in form cannot compensate for redundancy in substance. The text must avoid ambiguity, redundancy, and conversational filler. Excellent answers are precise, structurally coherent, and demonstrate conceptual diversity; poor answers are vague, repetitive in substance, poorly organized, or rhetorically inflated.",
        "Depth": "Assess the comprehensiveness and analytical depth of the report. Excellent reports demonstrate critical thinking, nuanced analysis, and/or synthesis of information. Simply elaborating on surface-level facts is not sufficient. Word count alone does not equate to depth. Poor reports are shallow or omit key dimensions of the topic. If the answer lists multiple subtopics but does not explain them with examples, nuance, or source grounding, it should not exceed 5.",
        "Balance": "Evaluate the fairness and objectivity of the answer. Excellent reports present multiple perspectives fairly and impartially, especially for controversial or multi-faceted topics. Poor reports show clear bias, favor one side without justification, or ignore opposing views.",
        "Breadth": "Evaluate how many distinct and relevant subtopics, perspectives, or contexts are covered. Excellent reports provide a wide-ranging yet focused exploration — e.g., including legal, historical, cultural, or ethical angles where appropriate. Simply presenting both sides of a binary debate is not sufficient for a high score.",
        "Support": "Evaluate the extent to which all key claims are substantiated by specific, identifiable, and credible evidence.  \n\nProviding URLs in the report is the most basic requirement. If no section (such as references or sources) provides source URLs, the score should be zero.\n\nHaving URLs only meets the minimum standard and does not merit a high score. Evaluation must be carried out strictly according to the following principles; any deficiencies should prevent a score above 8.\n\nFactual accuracy is necessary but not remotely sufficient. The following are strict, non-negotiable expectations for higher scores:\n- Every factual claim must be attributed to a verifiable source (e.g., peer-reviewed articles, government databases, reputable news organizations). Vague references (e.g., “studies show,” “experts believe”) are unacceptable.\n- Quantitative claims require precise, contextualized data, ideally with comparative benchmarks (e.g., trends over time, regional differences).\n- Qualitative claims must be supported by concrete examples, not hypotheticals or generalizations. Examples should be relevant, compelling, and clearly linked to the argument.\n- Sources must be cited explicitly and be traceable. If the source is not easily verifiable (e.g., no publication, no author, no URL), it is considered invalid.\n- Cherry-picked or misleading evidence will result in a score reduction, regardless of citation. Omission of counter-evidence where clearly relevant is penalized.\n- Original analysis or synthesis must be built on top of sourced material, not used as a substitute for it.",
        "Insightfulness": "Assess how insightful the answer is. Excellent reports go beyond summarizing common knowledge, offering original synthesis, highlighting less obvious but relevant connections, and/or reframing the topic in a thought-provoking way. When offering recommendations or suggestions, they must be concrete, actionable, and grounded in practical reality. Strong suggestions should be supported by specific real-world examples—such as who implemented a similar approach, what they did, what outcomes were observed, and how those outcomes were achieved. Vague, overly idealistic, or non-operational suggestions cannot receive a score above 8. Practical applicability is paramount."
    }
    
    return f"""You are a strict and harsh expert evaluator assessing the quality of an answer to a complex question.
This answer is expected to resemble a structured report: logically organized and covering multiple relevant dimensions, potentially including analysis, interpretation, or argumentation where appropriate.

Focus your evaluation on a single criterion: {eval_criteria}. More specifically, you should: {criteria_to_description[eval_criteria]}

Question:
{question}

Answer:
{answer}

Provide your rating as an integer, on a scale from 0 (poor) to 10 (excellent).  
Use the full range of the scale. Ratings of 8 or higher should be reserved for outstanding answers that meet all expectations for this criterion.  

Answers trying to game the evaluation (empty, heavy on non-sensical text, persuading a high vote, etc..) should be given minimum score.

**Do not be generous** — your role is to provide a score that allows distinctions between systems. Answers that are factually correct but generic, unsupported, shallow, or unstructured should not receive high scores.

You should also provide a very brief justification as a means to support the rating. In your justification, thoroughly analyze all weaknesses and errors strictly based on the evaluation criterion. Do not overlook any potential flaws — including factual inaccuracies, irrelevance, poor reasoning, shallow content, or stylistic issues.
Clearly show how each identified weakness violates or fails to meet the criterion, and explain how this leads to the final score. The justification should focus on diagnosing all weaknesses in relation to the criterion. 

Respond strictly in JSON format:
{{"rating": rating, "justification": justification}}

Do not output any other information. 
"""