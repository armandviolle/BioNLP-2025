import pandas as pd
import re
from openai import OpenAI
from typing import List
import re
from datasets import load_dataset
from i2b2_data_extraction import parse_xml_file
from utils import convert_xml_df
from tqdm import tqdm
# import os

# === Setup ===

# client = OpenAI(api_key="your-api-key-here")

# === Functions ===

def parse_gpt_output(output: str):
    # Try to find "Question:" or "question:" (case insensitive)
    question_match = re.search(r"[Qq]uestion:\s*(.*)", output)
    if question_match:
        return question_match.group(1).strip()
    
    # Fallback: If "Question" not found, just take the full output as the question
    return output.strip()


def get_all_few_shot_examples(df: pd.DataFrame) -> List[dict]:
    examples = []
    for note, group in df.groupby("note_excerpt"):
        clinician_questions = group["clinician_question"].dropna().unique()
        question = clinician_questions[0]
        examples.append({
            "note_excerpt": note,
            "question": question
            
        })
    return examples


def build_prompt(new_input: str, few_shot_examples: List[dict]) -> str:
    prompt_intro = """You are a clinical NLP assistant. 

Here are several examples of how to analyze the following examples and generate a clinical question from the note excerpt\n\n\n\n
"""

    few_shot_text = ""
    for idx, ex in enumerate(few_shot_examples):
        few_shot_text += (
            f"### Example {idx+1}\n\n\n"
            f"Input Note Excerpt:\n{ex['note_excerpt']}\n\n\n\n\n\n"
            f"Question: {ex['question']}\n\n\n\n"

        )

    task_text = (
        f"Now, your task is to analyze this new text and generate a question:\n\n\n\n"
        
        f"Note Excerpt:\n\n{new_input}\n\n"
        
        # f"Generate the following:\n\n"
        
        f"Question: <your generated question>"
        
    )


    return prompt_intro + few_shot_text + task_text


def remove_dates(text):
    if not text:
        return text

    # Pattern for dates
    date_pattern = r'\d{1,4}[-/]\d{1,2}[-/]\d{1,4}'

    # Pattern to match the full "Admission Date: date" or "Discharge Date: date" line
    admission_pattern = rf'(?i)Admission Date\s*[:\-]?\s*{date_pattern}'
    discharge_pattern = rf'(?i)Discharge Date\s*[:\-]?\s*{date_pattern}'

    # Remove the patterns
    text = re.sub(admission_pattern, '', text)
    text = re.sub(discharge_pattern, '', text)

    # Optional: clean extra spaces or empty lines if needed
    text = re.sub(r'\n\s*\n', '\n\n', text)  # remove multiple blank lines
    text = re.sub(r' +', ' ', text)  # collapse multiple spaces

    return text


# Helper function to normalize text
def normalize_text(text):
    text = str(text).lower().strip()
    text = re.sub(r'\s+', ' ', text)  # Collapse multiple spaces
    return text


def flatten_text(text):
    if not isinstance(text, str):
        return ""
    text = text.replace("\n", " ").replace("\r", " ").strip()
    text = re.sub(r"\s+", " ", text)  # normalize multiple spaces into one
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)  # remove space before punctuation
    text = text.lower()
    return text



# === Load Data ===

#i2b2 dataset
# I2B2-2012 data requires signing the n2nb2 license.

def generate_questions_summary(open_ai_key,i2b2_dataset_path,emrqa_dataset_hf_path = "Eladio/emrqa-msquad" ,mimic_3_dataset_hf_path = "Medilora/mimic_iii_diagnosis_anonymous"):

    client = OpenAI(api_key=open_ai_key)

    i2b2_data = parse_xml_file(i2b2_dataset_path)  

    i2b2_df = i2b2_data[0]

    i2b2_df = i2b2_df[['summary']]

    i2b2_labels = ['i2b2' for i in range(len(i2b2_df))]

    i2b2_df['source'] = i2b2_labels

    # Emrqa dataset

    ds = load_dataset(emrqa_dataset_hf_path)
    train_df = ds['train'].to_pandas()
    val_df = ds['validation'].to_pandas()
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)

    # Just in case it’s still named 'question' in train_df
    train_val_df.rename(columns={'question': 'query'}, inplace=True)

    # Extract just the first text value from the 'answers' dict
    train_val_df['answers'] = train_val_df['answers'].apply(lambda x: x['text'][0] if isinstance(x, dict) and x['text'] else None)

    # Keep only rows where answer is actually found in context
    train_val_df['answer_in_context'] = train_val_df.apply(lambda row: str(row['answers']).strip().lower() in str(row['context']).strip().lower(),axis=1)

    merged_df = train_val_df[train_val_df['answer_in_context']].copy()

    # Step 1: Filter contexts containing 'history' or 'hospital course'
    filtered_contexts = merged_df['context'].dropna().drop_duplicates()

    filtered_contexts = filtered_contexts[filtered_contexts.str.contains('history|hospital course', case=False, na=False)]

    emrqa_df = filtered_contexts.to_frame(name='summary')

    emrqa_labels = ['emrqa' for i in range(len(emrqa_df))]

    emrqa_df['source'] = emrqa_labels



    # Load the anonymized mimic-iii dataset

    dataset = load_dataset(mimic_3_dataset_hf_path)

    # Convert a specific split to a DataFrame, e.g., 'train'

    df = dataset['train'].to_pandas()

    df = df.rename(columns={'text': 'summary'})

    df = df[df['summary'].str.contains("discharge medications:|history of present illness:", case=False, na=False)]

    # Assume df is your DataFrame and it has a column named 'summary'

    # Step 1: Add a word count column
    df['word_count'] = df['summary'].apply(lambda x: len(str(x).split()))

    # Step 2: Filter rows with word count under 500
    filtered_df = df[df['word_count'] < 700]

    print("Filtered rows under 500 words:", len(filtered_df))

    # Step 3: Optionally bin word counts into ranges for diversity
    bins = [0, 250, 350, 450, 500]
    labels = ['short', 'medium', 'long', 'very_long']
    filtered_df['length_bin'] = pd.cut(filtered_df['word_count'], bins=bins, labels=labels)


    # Number of samples per bin
    samples_per_bin = 120

    # Sample 75 from each bin
    sampled_rows = (
        filtered_df.groupby('length_bin', group_keys=False)
        .apply(lambda x: x.sample(n=samples_per_bin, random_state=42))
    )

    # Final dataset
    sampled_rows = sampled_rows.drop_duplicates(subset='summary').reset_index(drop=True) 

    # BIO NLP 
    # sanity check : verify if it has any rows in common with the merged note cases

    df2 = convert_xml_df(xml_path = "dev/archehr-qa.xml", key_json_path = "dev/archehr-qa_key.json")

    # Step 1: Get unique note_excerpt per case_id
    unique_df2 = df2.drop_duplicates(subset='case_id')[['note_excerpt']].copy()
    # Apply normalization
    sampled_rows['normalized_summary'] = sampled_rows['summary'].apply(normalize_text)
    unique_df2['normalized_excerpt'] = unique_df2['note_excerpt'].apply(normalize_text)

    # Filter: remove any sampled_rows where normalized summary matches any normalized excerpt
    sampled_rows = sampled_rows[~sampled_rows['normalized_summary'].isin(unique_df2['normalized_excerpt'])].reset_index(drop=True)

    # Drop the helper column if you no longer need it
    sampled_rows = sampled_rows.drop(columns=['normalized_summary'])


    mimic3_df = sampled_rows[['summary']]

    mimic3_labels = ['mimic-iii' for i in range(len(mimic3_df))]

    mimic3_df['source'] = mimic3_labels

    #concatenate all the 3 datasets
    summaries_df = pd.concat([i2b2_df[['summary', 'source']], emrqa_df[['summary', 'source']],mimic3_df[['summary', 'source']]], ignore_index=True)

    summaries_df['summary'] = summaries_df['summary'].apply(remove_dates)


    #Create prompt examples using the note excerpts 

    few_shot_examples = get_all_few_shot_examples(df2)

    #Generate question from the concatenated discharge summaries 

    # === Loop Through Summaries ===

    all_rows = []
    case_id_counter = 1

    for i, row in tqdm(summaries_df.iterrows(), total=len(summaries_df)):
        note_excerpt = row["summary"]

        data_source = row['source']

        prompt = build_prompt(new_input=note_excerpt, few_shot_examples=few_shot_examples)


        # # Save prompt to file
        
        # prompt_save_dir = os.path.join(save_dir, "prompts")
        # os.makedirs(prompt_save_dir, exist_ok=True)  # create "prompts" folder if not exists

        # prompt_filename = f"prompt_case_{case_id_counter}.txt"
        # with open(os.path.join(prompt_save_dir, prompt_filename), "w", encoding="utf-8") as f:
        #     f.write(prompt)

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        output = response.choices[0].message.content.strip()
        question = parse_gpt_output(output)

        all_rows.append({
                "case_id": case_id_counter,
                "note_excerpt": note_excerpt,
                "generated_question": question,
                "source" : data_source
            })

        case_id_counter += 1

    # === Finalize and Save ===
    final_df = pd.DataFrame(all_rows)

    # Reorder columns
    final_df = final_df[[
        "case_id", "note_excerpt", "generated_question", "source" 
    ]]


    # # Save to Excel

    # save_dir = "question_generation/"
    # os.makedirs(save_dir, exist_ok=True)

    # final_df.to_excel(os.path.join(save_dir, "generated_questions.xlsx"), index=False)

    # # Preview
    # print(final_df)


    return final_df
