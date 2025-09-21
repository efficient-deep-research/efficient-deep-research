
def extract_final_information(output: str) -> str:
    pattern_info = "**Final Information**"
    if pattern_info in output:
        extracted_text = output.split(pattern_info)[-1].replace("\n", "").strip("```").strip()
    else:
        extracted_text = output

    return extracted_text
