import re
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Union
from bs4 import BeautifulSoup
from openai import OpenAI

# TODO: import useful functions from other python scripts
from utils import format_output, extract_sentences
from utils import responseGPT


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_key", default="", type=str)
    parser.add_argument("--data", default="./data/dev/archehr-qa.xml", type=str)
    parser.add_argument("--keys", default="./data/dev/archehr-qa_key.json", type=str)
    parser.add_argument("--model", default="gpt-4o-mini", type=str)
    parser.add_argument("--prompts_folder", default="", type=str)
    parser.add_argument("--res_path", default="", type=str)
    parser.add_argument("--date", default="", type=str)
    parser.add_argument("--n_seeds", default=10, type=int)
    parser.add_argument("--save_name", default="output", type=str)
    return parser.parse_args()


def load_system_prompts(prompts_folder: Union[Path, str]) -> dict:
    dic = {}
    with open(Path(prompts_folder) / "1-gen-answer.txt", "r") as f:
        dic["gen-answer"] = f.read()
    with open(Path(prompts_folder) / "2-retrieve-relevant-sentences.txt", "r") as f:
        dic["retrieve-relevant"] = f.read()
    # TODO
    with open(Path(prompts_folder) / "3-paraphrase.txt", "r") as f:
        dic["paraphrase"] = f.read()
    with open(Path(prompts_folder) / "4-format.txt", "r") as f:
        dic["format"] = f.read()
    with open(Path(prompts_folder) / "5-control-format.txt", "r") as f:
        dic["control-format"] = f.read()
    return dic


def formatUserPrompt(main_input: str, step: str, optional_input: str = None) -> str:
    if step == "gen-answer":
        ### input is case
        # Give the whole patient narrative and the clinical note as user prompt
        patient_narrative = main_input.find("patient_narrative").get_text(strip=True)
        clinical_note = main_input.find("note_excerpt").get_text(strip=True)
        return f"# Patient Narrative\n{patient_narrative}\n\n # Clinical Note\n{clinical_note}"
    elif step == "retrieve-relevant":
        ### input is case, optional input is previous answer
        # Give the previous answer and the single sentences of the note as user prompt
        patient_narrative = optional_input.find("patient_narrative").get_text(
            strip=True
        )
        single_sentences = optional_input.find("note_excerpt_sentences")
        return f"# Patient Narrative\n{patient_narrative}\n\n# Answer\n{main_input}\n\n# Sentences\n{single_sentences}"
    elif step == "paraphrase":
        ### input is case, optional input is predicted labels
        # Give the patient narrative, clinician question and single sentences with essential labels (and supplementary?)
        patient_narrative = main_input.find("patient_narrative").get_text(strip=True)
        clinician_question = main_input.find("clinician_question").get_text(strip=True)
        single_sentences, sentences_prompt = extract_sentences(
            case=main_input, list_ids=optional_input
        )
        return f"# Patient Narrative\n{patient_narrative}\n\n# Clinician Question\n{clinician_question}\n\n# Relevant Sentences\n{sentences_prompt}"
    elif step == "format":
        ### input is previous answer, optional input is (case, ids)
        # Give the previous answer, patient narrative and single sentences
        case, ids = optional_input
        patient_narrative = case.find("patient_narrative").get_text(strip=True)
        single_sentences, sentences_prompt = extract_sentences(case=case, list_ids=ids)
        return f"# Answer\n{main_input}\n\n# Patient Narrative\n{patient_narrative}\n\n# Sentences\n{sentences_prompt}"


def main(args: argparse.Namespace) -> None:
    client = OpenAI(api_key=args.client)

    system_prompts = load_system_prompts(prompts_folder=args.prompts_folder)

    with open(args.data, "r") as dat:
        data = BeautifulSoup(dat, "xml")
    with open(args.keys, "r") as k:
        keys = json.load(k)

    cases = [case for case in data.find_all("case")]
    # labelled_cases = format_input(cases=cases, relevance_keys=keys)
    answers = []
    for case in cases:
        case_id = case.get("id")
        sentence_ids = []

        # ===== GENERATION =====

        # if case_id == "1":
        print(f"\n======== CASE ID {case_id}=========\n")
        seeds = np.random.choice(
            range(0, 43), size=args.n_seeds, replace=False
        ).tolist()
        print(f"SEEDS: {seeds}")
        for s in seeds:
            user_prompt1 = formatUserPrompt(main_input=case, step="gen-answer")
            answer1 = responseGPT(
                client=client,
                system_prompt=system_prompts["gen-answer"],
                user_prompt=formatUserPrompt(main_input=case, step="gen-answer"),
                model_name=args.model,
                seed=s,
            )
            print(f"ANSWER\n{answer1}\n")
            user_prompt2 = formatUserPrompt(
                main_input=answer1, step="retrieve-relevant", optional_input=case
            )
            answer2 = responseGPT(
                client=client,
                system_prompt=system_prompts["retrieve-relevant"],
                user_prompt=formatUserPrompt(
                    main_input=answer1, step="retrieve-relevant", optional_input=case
                ),
                model_name=args.model,
                seed=s,  # use same seed?
            )
            # if s == seeds[0]:
            #     print("========= USER PROMPTS =========")
            #     print(user_prompt1)
            #     print(user_prompt2)
            #     print()
            # print(f"CITATIONS\n{answer2}\n")
            # print(f"IDs")
            # print(re.findall(r'\b\d{1,2}\b', answer2.splitlines()[-1]))
            # print()
            ids = re.findall(r"\b\d{1,2}\b", answer2.splitlines()[-1])
            print(ids)
            sentence_ids += ids

        # ===== CLASSIFICATION =====

        # Classify the cited sentences
        #   - if cited more than 80%, then essential
        #   - if cited between 50 and 80%, then supplementary
        #   - else not relevant
        single_ids = list(set(sentence_ids))
        print(single_ids)
        freq = {}
        for id_ in single_ids:
            freq[id_] = sentence_ids.count(id_) / args.n_seeds
        pred_essentials = [id_ for id_ in list(freq.keys()) if freq[id_] >= 0.75]
        print("\n====== predicted essentials ======")
        print(freq)
        print(pred_essentials)

        # ===== PARAPHRASING =====

        # Answer to the patient narrative by paraphrasing the essential (and supplementary) responses
        paraphrases1 = responseGPT(
            client=client,
            system_prompt=system_prompts["paraphrase"],
            user_prompt=formatUserPrompt(
                main_input=case, step="paraphrase", optional_input=pred_essentials
            ),
            model_name=args.model,
            # FIX SEED ?
        )
        print("=== paraphrase1 ===")
        print(paraphrases1)

        # Force response format #1 (prompt)
        paraphrases2 = responseGPT(
            client=client,
            system_prompt=system_prompts["format"],
            user_prompt=formatUserPrompt(
                main_input=paraphrases1,
                step="format",
                optional_input=(case, pred_essentials),
            ),
            model_name=args.model,
            # FIX SEED ?
        )
        print("=== paraphrase2 ===")
        print(paraphrases2)
        # print(paraphrases2.split(" "))
        print(len(paraphrases2.split(" ")))

        # Force response format #1 (prompt)
        paraphrases3 = responseGPT(
            client=client,
            system_prompt=system_prompts["control-format"],
            user_prompt=formatUserPrompt(
                main_input=paraphrases2,
                step="format",
                optional_input=(case, pred_essentials),
            ),
            model_name=args.model,
            # FIX SEED ?
        )
        print("=== paraphrase2 ===")
        print(paraphrases3)
        # print(paraphrases3.split(" "))
        print(len(paraphrases3.split(" ")))

        # Force response format #3 (post-process)
        final_answer = format_output(answer=paraphrases3)
        print("=== final answer ===")
        print(final_answer)
        print(len(final_answer.split(" ")))
        answers.append({"case_id": case_id, "answer": final_answer})

    with open(Path(args.res_path) / f"{args.save_name}.json", "w") as res:
        # Try to find a better json formatting method when saving the results
        json.dump(answers, res)


if __name__ == "__main__":
    main()
