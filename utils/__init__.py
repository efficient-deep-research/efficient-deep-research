def extract_answer(output: str) -> str:
    extracted_text = ""
    # Extract content after **Final Information** or **Modified Reasoning Steps**
    # pattern_info = "\n**Final Information**"
    # pattern_step = "\n**Modified Reasoning Steps**"
    pattern_info = "**Final Information**"
    pattern_step = "**Modified Reasoning Steps**"
    if pattern_info in output:
        extracted_text = output.split(pattern_info)[-1].replace("\n", "").strip("```").strip()
    elif pattern_step in output:
        extracted_text = output.split(pattern_step)[-1].strip("```").strip()
    else:
        # extracted_text = "No helpful information found."
        extracted_text = output

    return extracted_text
