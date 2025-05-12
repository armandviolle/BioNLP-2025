import os, sys, json, argparse, bs4
from pathlib import Path
from typing import Union
import numpy as np
from bs4 import BeautifulSoup
from openai import OpenAI


def str2bool(v):
    if v.lower() in ('true', '1'):
        return True
    elif v.lower() in ('false', '0'):
        return False

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_key", default="", type=str)
    parser.add_argument("--data", default="./data/dev/archehr-qa.xml", type=str)
    parser.add_argument("--keys", default="./data/dev/archehr-qa_key.json", type=str)
    parser.add_argument("--prompts_folder", default="", type=str)
    parser.add_argument("--model", default="gpt-4o-mini", type=str)
    parser.add_argument("--res_path", default="", type=str)
    parser.add_argument("--date", default="", type=str)
    parser.add_argument("--n_seeds", default=5, type=int)
    parser.add_argument("--save_name", default="", type=str)
    parser.add_argument("--mode", default="", type=str)
    return parser.parse_args()



def format_input(
    cases: list, 
    relevance_keys: list, 
) -> list:
    res_cases = []
    for case in cases:
        case_id = case['id']
        case_dict = {
            'case_id': case_id, 
            'full_case': case,
            'patient_question': case.find('patient_question').get_text(strip=True), 
            'clinician_question': case.find('clinician_question').get_text(strip=True), 
            'sentences': {'essential': [], 'supplementary': [], 'not-relevant': []}
        }
        sentences = case.find('note_excerpt_sentences')
        for sentence in sentences.find_all('sentence'):
            senID = sentence['id']
            # if relevance_keys[int(case_id)-1]['answers'][int(senID)]['relevance'] in ['essential', 'supplementary']:
            senRelevance = relevance_keys[int(case_id)-1]['answers'][int(senID)-1]['relevance']
            case_dict['sentences'][senRelevance].append({
                'sentence_id': senID, 
                'sentence_text': sentence.get_text(strip=True), 
                'sentence_xml': f"<sentence id={senID}>\n{sentence.get_text(strip=True)}\n</sentence>"
            })
        res_cases.append(case_dict)
    return res_cases


def find_essentials(
    case_id: str,
    keys: dict,
) -> list:
    out = []
    for i in range(len(keys)):
        if keys[i]['case_id'] == case_id:
            answers = keys[i]['answers']
            for answer in answers:
                if answer['relevance'] == "essential":
                    out.append(answer['sentence_id'])
            break
    return out


def extract_sentences(
    case_id: str, 
    list_ids: list, 
    cases: BeautifulSoup
) -> list:
    case = soup.find(lambda tag: tag.name=='case' and tag.has_attr('id') and tag['id']==case_id)
    out = [case.find('patient_question').get_text(strip=True)]
    sentences = case.find('note_excerpt_sentences')
    for sentence in sentences.find_all('sentence'):
        if sentence['id'] in list_ids:
            out.append(sentence.get_text(strip=True))
    return "\n".join(out)



import re
def format_output(output):
    new_output = []
    pattern = r'.*?\|\d+(?:,\d+)*\|'
    for res_case in output:
        parts = re.findall(pattern, output['answer'])
        parts = [part.strip() for part in parts]
        new_output.append({'case_id': str(res_case['case_id']), 'answer': "\n".join(parts)})
    return new_output

def format_output_answer(original_answer):
    pattern = r'.*?\|\d+(?:,\d+)*\|'
    parts = re.findall(pattern, original_answer)
    parts = [part.strip() for part in parts]
    return "\n".join(parts)



def zeroShotGPT(
    client,
    system_prompt: str, 
    user_prompt: str, 
    model_name: str,
    seed: int = 42
) -> str:
    completion = client.chat.completions.create(
        model=model_name, 
        messages=[
            {"role": "system", "content": system_prompt}, 
            {"role": "user", "content": user_prompt}, 
        ],
        seed=seed
    )
    return completion.choices[0].message.content



def formatInferencePrompt(
    case: bs4.element.Tag,
) -> str: 
    return f"{case.find('patient_narrative')}\n{case.find('patient_question')}\n{case.find('clinician_question')}\n{case.find('note_excerpt_sentences')}\n"


def fewShotGPT(
    client,
    system_prompt: str, 
    user_assistant_prompts: list,
    user_inference_prompt: str, 
    model_name: str,
    seed: int = 42
) -> str:
    messages = [{"role": "system", "content": system_prompt}]
    for i in range(len(user_assistant_prompts)):
        messages.append({"role": "user", "content": user_assistant_prompts[i]['user']})
        messages.append({"role": "assistant", "content": user_assistant_prompts[i]['assistant']})
    messages.append({"role": "user", "content": user_inference_prompt})
    completion = client.chat.completions.create(
        model=model_name, 
        messages=messages, 
        seed=seed
    )
    return completion.choices[0].message.content



def formatParaphrasePrompt(case: dict) -> str:
    essential_sentences = case['sentences']['essential']
    sentences_prompt = "\n".join([sentence['sentence_xml'] for sentence in essential_sentences])
    question = case['patient_question']
    return f"{sentences_prompt}\n<question>\n{question}\n</question>\n"

def generateUserAssistantPrompts(
    client,
    labelled_cases: list,
    system_paraphrasing_prompt: str, 
    model_name: str, 
    seed: int = 42
) -> list:
    responses = []
    for case in labelled_cases: 
        user_paraphrasing_input = formatParaphrasePrompt(case=case)
        # if case['case_id'] == "1":
        #     print(f"\nSYSTEM PARAPHRASING PROMPT\n{system_paraphrasing_prompt}\n")
        #     print(f"\nUSER PARAPHRASING PROMPT\n{user_paraphrasing_prompt}\n")
        #     print(f"\nUSER PARAPHRASING INPUT (case 1)\n{user_paraphrasing_input}\n")
        responses.append({
            'case_id': case['case_id'], 
            'user': formatInferencePrompt(case=case['full_case']), 
            'assistant': format_output_answer(zeroShotGPT(
                client,
                system_prompt=system_paraphrasing_prompt, 
                user_prompt=user_paraphrasing_input, 
                model_name=model_name, 
                seed=seed
            ))
        })
    return responses



def mainFewShot(args):

    client = OpenAI(api_key=args.client_key)
    
    with open(args.data, 'r') as dat:
        data = BeautifulSoup(dat, 'xml')
    with open(args.keys, 'r') as k:
        keys = json.load(k)
    with open(Path(args.prompts_folder) / "zero-shot.txt", 'r') as sys:
        system_prompt = sys.read()
    with open(Path(args.prompts_folder) / "zero-shot-paraphrasing.txt", 'r') as sys_par:
        system_paraphrasing_prompt = sys_par.read()
    
    paraphrases_dir = Path(args.res_path) / "paraphrases"
    os.makedirs(paraphrases_dir, exist_ok=True)
    answers_dir = Path(args.res_path) / "answers"
    os.makedirs(answers_dir, exist_ok=True)

    cases = [case for case in data.find_all('case')]
    if os.path.exists(Path(paraphrases_dir) / f"paraphrases_{args.model}.json"):
        print(f"\nUsing existing paraphrases at {Path(paraphrases_dir) / f"paraphrases_{args.model}.json"}\n")
        with open(Path(paraphrases_dir) / f"paraphrases_{args.model}.json", 'r') as res_para:
            paraphrases = json.load(res_para)
    else:
        print("\nGenerating paraphrases")
        labelled_cases = format_input(cases=cases, relevance_keys=keys)
        paraphrases = generateUserAssistantPrompts(
            client=client,
            labelled_cases=labelled_cases, 
            system_paraphrasing_prompt=system_paraphrasing_prompt, 
            model_name=args.model
        )
        with open(Path(paraphrases_dir) / f"paraphrases_{args.model}.json", 'w') as res_para:
            json.dump(paraphrases, res_para)
        print(f"Using generated paraphrases saved at {Path(paraphrases_dir) / f"paraphrases_{args.model}.json"}\n")
    
    for seed in range(args.n_seeds):
        case_id = 1
        answers = []
        for i in range(len(cases)):
            all_examples = [ex for ex in paraphrases if ex['case_id']!=str(case_id)] # TODO: choose 5 randomly
            selected_ids = np.random.choice([ex['case_id'] for ex in all_examples], size=5, replace=False)
            print(f"Selected example-cases for case {case_id}: {selected_ids}")
            used_examples = [ex for ex in all_examples if ex['case_id'] in selected_ids]
            inference_case = cases[i]
            print(f"Few-shotting case {inference_case['id']}")
            answer = fewShotGPT(
                client=client,
                system_prompt=system_prompt, 
                user_assistant_prompts=used_examples,
                user_inference_prompt=formatInferencePrompt(case=inference_case), 
                model_name=args.model,
                seed=seed
            )
            answers.append({'case_id': str(case_id), 'answer': format_output_answer(answer)})
            case_id += 1
        print(f"\nEnd of few-shot on seed {seed}, saving results.\n")
        with open(Path(answers_dir) / f"answers_{args.model}_seed{seed}.json", 'w') as res:
            json.dump(answers, res)



def mainZeroShot(args):

    client = OpenAI(api_key=args.client_key)
    
    with open(args.data, 'r') as dat:
        data = BeautifulSoup(dat, 'xml')
    with open(args.keys, 'r') as k:
        keys = json.load(k)
    with open(Path(args.prompts_folder) / "zero-shot.txt", 'r') as sys:
        system_prompt = sys.read()
    with open(Path(args.prompts_folder) / "zero-shot-paraphrasing.txt", 'r') as sys_par:
        system_paraphrasing_prompt = sys_par.read()
    
    answers_dir = Path(args.res_path) / "answers-zero"
    os.makedirs(answers_dir, exist_ok=True)

    cases = [case for case in data.find_all('case')]
    for seed in range(args.n_seeds):
        print(f"\nWorking on seed {seed}\n")
        case_id = 1
        answers = []
        for i in range(len(cases)):
            inference_case = cases[i]
            print(f"Few-shotting case {inference_case['id']}")
            answer = zeroShotGPT(
                client=client,
                system_prompt=system_prompt, 
                user_prompt=formatInferencePrompt(case=inference_case),
                model_name=args.model, 
                seed=seed,
            )
            answers.append({'case_id': str(case_id), 'answer': format_output_answer(answer)})
            case_id += 1
        print(f"\nEnd of few-shot on seed {seed}, saving results.\n")
        with open(Path(answers_dir) / f"answers_{args.model}_seed{seed}.json", 'w') as res:
            json.dump(answers, res)


def main():
    args = parse()
    if args.mode == "few-shot":
        mainFewShot(args=args)
    elif args.mode == "zero-shot":
        mainZeroShot(args=args)


if __name__ == "__main__":
    main()