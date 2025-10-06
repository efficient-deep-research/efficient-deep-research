import re
from collections import defaultdict


def extract_incorrect_markers(input_string):
    """
    Extract incorrect markers from the input string. Correct markers are in the form of <|keyword|>.

    Args:
        input_string (str): Input string.
    Returns:
        List[str]: List of incorrect markers found in the input string.
    """

    standard_keywords = ["begin_search_query", "end_search_query", "begin_search_result", "end_search_result"]

    # Create regex pattern to match standard keywords
    pattern = re.compile("(" + "|".join(map(re.escape, standard_keywords)) + ")")

    incorrect_markers = []

    for match in pattern.finditer(input_string):
        keyword = match.group()
        start, end = match.start(), match.end()

        # Check if the preceding two characters are <|
        prefix_ok = start >= 2 and input_string[start - 2 : start] == "<|"
        # Check if the following two characters are |>
        suffix_ok = end + 2 <= len(input_string) and input_string[end : end + 2] == "|>"
        if not (prefix_ok and suffix_ok):
            keyword = input_string[start - 2 : end + 2]
            incorrect_markers.append(keyword)

    return incorrect_markers


def detect_variant_markers(input_string):
    """
    Detect whether variant forms of markers exist in the input string.

    Args:
        input_string (str): Input string.

    Returns:
        bool: Whether variant markers exist or think tags are used excessively. Returns True if issues are detected, otherwise False.
    """

    variant_markers = extract_incorrect_markers(input_string)

    think_count = input_string.count("</think>")
    has_think_issue = think_count > 2

    return (len(variant_markers) > 0) or has_think_issue


def is_valid_history(history):
    """
    Detect whether the history list conforms to valid cases.

    Args:
        history (list): Input history list.

    Returns:
        bool: Returns True if it conforms to valid cases, otherwise returns False.
    """

    SPECIAL_TOKENS = {
        "<|begin_search_query|>",
        "<|end_search_query|>",
        "<|begin_search_result|>",
        "<|end_search_result|>",
        "</think>",
    }

    for hid, his in enumerate(history):
        full_text = his.strip()

        # Case 1: Only one </think> and it's the last element
        if "</think>" in his:
            if hid == len(history) - 1 and history[-1].count("</think>") == 1:
                for token in SPECIAL_TOKENS:
                    if token != "</think>" and token in full_text:
                        return False
            else:
                return False

        # Case 2: Starts with <|begin_search_result|> and ends with <|end_search_result|>, no other special markers in between
        elif full_text.startswith("<|begin_search_result|>") and full_text.endswith("<|end_search_result|>"):
            middle_content = full_text[len("<|begin_search_result|>") : -len("<|end_search_result|>")].strip()
            if any(token in middle_content for token in SPECIAL_TOKENS):
                return False

        # Case 3: Ends with <|end_search_query|>, contains exactly one <|begin_search_query|> in the middle, no other special tokens
        elif full_text.endswith("<|end_search_query|>"):
            middle_content = full_text[: -len("<|end_search_query|>")].strip()
            if middle_content.count("<|begin_search_query|>") == 1:
                # Remove <|begin_search_query|> and check for other special markers
                middle_without_begin = middle_content.replace("<|begin_search_query|>", "").strip()
                if any(token in middle_without_begin for token in SPECIAL_TOKENS):
                    return False
            else:
                return False

        else:
            return False
    return True
