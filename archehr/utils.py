import os
import re
import json
from pathlib import Path
from openai import OpenAI
from typing import Any, Dict, List
from xml.etree import ElementTree as ET
import pandas as pd


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


def convert_xml_df(xml_path, key_json_path):
    # Load and parse the XML file

    tree = ET.parse(xml_path)
    root = tree.getroot()

    cases_data = []

    # Iterate through <case> elements
    for case in root.findall("case"):
        case_id = case.attrib["id"]
        patient_narrative = case.findtext("patient_narrative", default="").strip()
        clinician_question = case.findtext("clinician_question", default="").strip()

        # Extract phrases
        phrases = []
        for phrase in case.findall(".//patient_question/phrase"):
            phrases.append(phrase.text.strip())

        # Extract note sentences
        note_sentences = []
        for sentence in case.findall(".//note_excerpt_sentences/sentence"):
            sent_id = sentence.attrib["id"]
            text = sentence.text.strip() if sentence.text else ""
            note_sentences.append({
                "case_id": case_id,
                "sentence_id": sent_id,
                "ref_excerpt": text,
            })

        # Extract full note excerpt
        note_excerpt = case.findtext("note_excerpt", default="").strip()

        cases_data.append({
            "case_id": case_id,
            "patient_narrative": patient_narrative,
            "patient_question_phrases": phrases,
            "clinician_question": clinician_question,
            "note_excerpt": note_excerpt,
            "note_sentences": note_sentences,
        })

    # Flatten note_sentences into a DataFrame
    note_sentences_df = pd.DataFrame([
        s for case in cases_data for s in case["note_sentences"]
    ])

    # Build the main case-level DataFrame
    cases_df = pd.DataFrame([
        {
            "case_id": c["case_id"],
            "patient_narrative": c["patient_narrative"],
            "patient_question_phrases": c["patient_question_phrases"],
            "clinician_question": c["clinician_question"],
            "note_excerpt": c["note_excerpt"],
        }
        for c in cases_data
    ])

    # Load the key.json file

    with open(key_json_path, "r") as f:
        key_data = json.load(f)

    # Convert key.json to DataFrame
    relevance_labels = []
    for entry in key_data:
        case_id = entry["case_id"]
        for ans in entry["answers"]:
            relevance_labels.append({
                "case_id": case_id,
                "sentence_id": ans["sentence_id"],
                "relevance": ans["relevance"],
            })

    relevance_df = pd.DataFrame(relevance_labels)

    # Merge sentence-level data with relevance labels
    note_with_labels_df = pd.merge(
        note_sentences_df, relevance_df, on=["case_id", "sentence_id"], how="left"
    )

    # Add full note_excerpt to each row
    note_with_labels_df = pd.merge(
        note_with_labels_df,
        cases_df[["case_id", "note_excerpt"]],
        on="case_id",
        how="left",
    )

    # Merge all_cases and notes_with_excerpt
    merged_df = pd.merge(
        note_with_labels_df.drop(
            columns=["note_excerpt"]
        ),  # drop to avoid duplicate column
        cases_df,  # includes note_excerpt
        on="case_id",
        how="left",
    )

    # Reorder columns so note_excerpt comes right after case_id
    cols = list(merged_df.columns)
    cols.remove("note_excerpt")
    case_id_index = cols.index("case_id")
    cols.insert(case_id_index + 1, "note_excerpt")
    merged_df = merged_df[cols]

    return merged_df
