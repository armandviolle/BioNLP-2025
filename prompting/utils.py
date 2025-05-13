import os, re, sys, json, argparse, bs4
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



# def extract_sentences(
#     case_id: str, 
#     list_ids: list, 
#     cases: BeautifulSoup
# ) -> list:
#     case = soup.find(lambda tag: tag.name=='case' and tag.has_attr('id') and tag['id']==case_id)
#     out = [case.find('patient_question').get_text(strip=True)]
#     sentences = case.find('note_excerpt_sentences')
#     for sentence in sentences.find_all('sentence'):
#         if sentence['id'] in list_ids:
#             out.append(sentence.get_text(strip=True))
#     return "\n".join(out)


def extract_sentences(
    case: str, 
    list_ids: list
) -> list:
    out = {}
    prompt = ""
    sentences = case.find('note_excerpt_sentences')
    for sentence in sentences.find_all('sentence'):
        if sentence['id'] in list_ids:
            out[sentence['id']] = sentence.get_text(strip=True)
            prompt += f"## sentence id: {sentence['id']}\n{sentence.get_text(strip=True)}\n"
    return out, prompt



def format_output(answer):
    pattern = r'.*?\|\d+(?:,\d+)*\|'
    parts = re.findall(pattern, answer)
    parts = [part.strip() for part in parts]
    return "\n".join(parts)



def responseGPT(
    client: client, 
    system_prompt: str, 
    user_prompt: str, 
    model_name: str,
    seed: int = 13
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
