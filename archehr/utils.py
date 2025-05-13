import os
import re
import json
from pathlib import Path
from openai import OpenAI
from typing import Any, Dict, List
from xml.etree import ElementTree as ET


def str2bool(v):
    if v.lower() in ("true", "1"):
        return True
    elif v.lower() in ("false", "0"):
        return False


def load_data(data_path: str) -> List[Dict[str, Any]]:
    """
    Load the data of the Archehr dataset.
    Args:
        data_path (str): Path to the dataset.
    Returns:
        Dict[str, Any]: Dictionary containing the data. Keys are the case ids,
        values are dictionaries containing the narrative, patient question,
        clinical question, and sentences.
        Senteces are tuples containing the sentence id, sentence text, and
        relevance.
    """
    case_file = "archehr-qa.xml"
    json_file = "archehr-qa_key.json"
    dev_mode = False

    # Check if the data path is valid
    if not os.path.isdir(data_path):
        raise ValueError(f"Invalid directory: {data_path}")

    if case_file not in os.listdir(data_path):
        raise ValueError(f"Missing {case_file} in {data_path}")

    if json_file in os.listdir(data_path):
        print(f"{json_file} in {data_path}. It must be a dev file.")
        dev_mode = True

    # Load the xml file
    tree = ET.parse(Path(data_path) / case_file)
    root = tree.getroot()

    # Load the json file
    with open(Path(data_path) / json_file, "r") as f:
        labels = json.load(f)

    # Transform the xml data into a dictionary
    data = {}
    for c in root.findall("case"):
        data[c.get("id")] = {
            "case_id": c.get("id"),
            "narrative": c.find("patient_narrative").text,
            "patient_question": c.find("patient_question").find("phrase").text,
            "clinician_question": c.find("clinician_question").text,
            "clinical_note": "\n".join([
                sentence.text.strip()
                for sentence in c.find("note_excerpt_sentences").findall("sentence")
            ]),
            "unlabeled_sentences": [
                {"sentence_id": j + 1, "text": sentence.text.strip()}
                for j, sentence in enumerate(
                    c.find("note_excerpt_sentences").findall("sentence")
                )
            ],
            "sentences": [
                (j, sentence.text.strip())
                for j, sentence in enumerate(
                    c.find("note_excerpt_sentences").findall("sentence")
                )
            ],
        }
    if dev_mode:
        for c, label in zip(root.findall("case"), labels):
            data[c.get("id")]["labeled_sentences_lenient"] = [
                {
                    "sentence_id": j + 1,
                    "text": sentence.text.strip(),
                    "label": "essential"
                    if answer["relevance"] in ["essential", "supplementary"]
                    else "not-relevant",
                }
                for j, (sentence, answer) in enumerate(
                    zip(
                        c.find("note_excerpt_sentences").findall("sentence"),
                        label["answers"],
                        strict=True,
                    )
                )
            ]
            data[c.get("id")]["labeled_sentences_strict"] = [
                {
                    "sentence_id": j + 1,
                    "text": sentence.text.strip(),
                    "label": "essential"
                    if answer["relevance"] == "essential"
                    else "non-essential",
                }
                for j, (sentence, answer) in enumerate(
                    zip(
                        c.find("note_excerpt_sentences").findall("sentence"),
                        label["answers"],
                        strict=True,
                    )
                )
            ]
    return data


def format_input(
    cases: list,
    relevance_keys: list,
) -> list:
    res_cases = []
    for case in cases:
        case_id = case["id"]
        case_dict = {
            "case_id": case_id,
            "full_case": case,
            "patient_question": case.find("patient_question").get_text(strip=True),
            "clinician_question": case.find("clinician_question").get_text(strip=True),
            "sentences": {"essential": [], "supplementary": [], "not-relevant": []},
        }
        sentences = case.find("note_excerpt_sentences")
        for sentence in sentences.find_all("sentence"):
            senID = sentence["id"]
            # if relevance_keys[int(case_id)-1]['answers'][int(senID)]['relevance'] in ['essential', 'supplementary']:
            senRelevance = relevance_keys[int(case_id) - 1]["answers"][int(senID) - 1][
                "relevance"
            ]
            case_dict["sentences"][senRelevance].append({
                "sentence_id": senID,
                "sentence_text": sentence.get_text(strip=True),
                "sentence_xml": f"<sentence id={senID}>\n{sentence.get_text(strip=True)}\n</sentence>",
            })
        res_cases.append(case_dict)
    return res_cases


def find_essentials(
    case_id: str,
    keys: dict,
) -> list:
    out = []
    for i in range(len(keys)):
        if keys[i]["case_id"] == case_id:
            answers = keys[i]["answers"]
            for answer in answers:
                if answer["relevance"] == "essential":
                    out.append(answer["sentence_id"])
            break
    return out


def extract_sentences(case: str, list_ids: list) -> list:
    out = {}
    prompt = ""
    sentences = case.find("note_excerpt_sentences")
    for sentence in sentences.find_all("sentence"):
        if sentence["id"] in list_ids:
            out[sentence["id"]] = sentence.get_text(strip=True)
            prompt += (
                f"## sentence id: {sentence['id']}\n{sentence.get_text(strip=True)}\n"
            )
    return out, prompt


def format_output(answer):
    pattern = r".*?\|\d+(?:,\d+)*\|"
    parts = re.findall(pattern, answer)
    parts = [part.strip() for part in parts]
    return "\n".join(parts)


def responseGPT(
    client: OpenAI,
    system_prompt: str,
    user_prompt: str,
    model_name: str,
    seed: int = 13,
) -> str:
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        seed=seed,
    )
    return completion.choices[0].message.content
