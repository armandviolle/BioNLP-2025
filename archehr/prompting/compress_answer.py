import json
import re
import os
from openai import OpenAI
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize
from archehr import BASE_DIR

# Load API key
client = OpenAI(
    # This is the default and can be omitted
    api_key="YOUR_API_KEY",
)

nltk.download("punkt")
nltk.download("punkt_tab")

MAX_WORDS = 75


def extract_sentence_citation_pairs(answer):
    pattern = r"(.*?)\|([\d,\-]+)\|"
    matches = re.findall(pattern, answer, re.DOTALL)
    results = []
    for text, citation_str in matches:
        sentences = sent_tokenize(text.strip())
        citations_raw = [c.strip() for c in citation_str.split(",")]
        citations = []
        for c in citations_raw:
            if "-" in c:
                start, end = map(int, c.split("-"))
                citations.extend([str(i) for i in range(start, end + 1)])
            else:
                citations.append(c)
        for sentence in sentences:
            if sentence:
                results.append((sentence.strip(), citations))
    return results


def group_sentences_by_citation(pairs):
    grouped = []
    current_group = []
    current_citations = None
    for sent, cits in pairs:
        if current_citations is None:
            current_group.append(sent)
            current_citations = cits
        elif cits == current_citations:
            current_group.append(sent)
        else:
            grouped.append((" ".join(current_group), current_citations))
            current_group = [sent]
            current_citations = cits
    if current_group:
        grouped.append((" ".join(current_group), current_citations))
    return grouped


def word_count(text):
    return len(text.split())


def condense_text(text, word_limit=20):
    prompt = (
        f"Summarize the following medical explanation into a shorter version preserving key clinical information. "
        f"Keep it under {word_limit} words:\n\n{text}"
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful medical summarization assistant.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI API error on text: {text[:50]}... =>", e)
        return text  # fallback to original


def process_case(case):
    case_id = case["case_id"]
    original_answer = case["answer"]
    pairs = extract_sentence_citation_pairs(original_answer)
    grouped = group_sentences_by_citation(pairs)

    new_answer_lines = []
    total_words = 0
    used_citations = set()

    for group_text, citations in grouped:
        max_words_remaining = MAX_WORDS - total_words
        if max_words_remaining <= 0:
            break

        summarized = condense_text(group_text, word_limit=max_words_remaining).strip()

        if summarized and summarized[-1] not in ".!?":
            summarized += "."

        num_words = word_count(summarized)
        if total_words + num_words <= MAX_WORDS:
            citation_str = ",".join(citations)
            new_answer_lines.append(f"{summarized} |{citation_str}| \n")
            total_words += num_words
            used_citations.update(citations)
        else:
            break

    # Get all original citations
    all_citations = {c for _, cits in pairs for c in cits}

    # If some citations were missed, add a final line to include them
    missed_citations = all_citations - used_citations
    if missed_citations:
        print(f"[case {case_id}] Adding missed citations: {missed_citations}")
        citation_str = ",".join(sorted(missed_citations))
        new_answer_lines.append(f"Additional supporting evidence. |{citation_str}| \n")

    # Final fallback if everything failed
    if not new_answer_lines:
        print(
            f"[Warning] Case {case_id} has no citations after processing. Adding fallback."
        )
        if pairs:
            sentence, citations = pairs[0]
            citation_str = ",".join(citations or ["unknown"])
            fallback_sent = (
                sentence
                if word_count(sentence) <= MAX_WORDS
                else condense_text(sentence, word_limit=MAX_WORDS)
            )
            new_answer_lines.append(f"{fallback_sent.strip()} |{citation_str}| \n")
        else:
            new_answer_lines.append("No valid content found. |unknown| \n")

    return {"case_id": case_id, "answer": "".join(new_answer_lines)}


def process_submission(input_path, output_path):
    with open(input_path, "r") as infile:
        cases = json.load(infile)

    updated = [process_case(case) for case in tqdm(cases)]

    with open(output_path, "w") as outfile:
        json.dump(updated, outfile, indent=4)


if __name__ == "__main__":
    results_path = BASE_DIR / "data" / "submission_files" / "dev" / "few_shot"
    for i in os.listdir(results_path):
        original_submission = results_path.joinpath(i)
        print("Processing file: ", original_submission)
        cleaned_submission_path = (
            BASE_DIR / "data" / "submission_files" / "dev" / "few_shot_cleaned"
        ).joinpath(i)
        process_submission(original_submission, cleaned_submission_path)
