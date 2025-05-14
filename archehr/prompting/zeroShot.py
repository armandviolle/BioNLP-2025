import pandas as pd
import re
from utils import convert_xml_df
from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm


# # === Load OpenAI key ===

open_ai_api_key = "YOUR_API_KEY"

# === Load sentence-level clinical notes ===

df = convert_xml_df(xml_path = "dev/archehr-qa.xml", key_json_path = "dev/archehr-qa_key.json")

df["ref_excerpt"] = df["ref_excerpt"].fillna("").astype(str)

# Re-label for binary classification (essential vs not-relevant)
df["binary_relevance"] = df["relevance"].apply(lambda x: "essential" if x == "essential" else "not-relevant")



# === Define which question to use ===

# question_column = "clinician_question"
question_column = "patient_narrative"

# Group clinical questions and relevant sentences by case
cases = df.groupby("case_id").agg({
    question_column: "first"
}).reset_index()

# Optionally filter out cases with no valid question
cases = cases[cases[question_column].notna()]

def extract_cited_ids(answer_text):
    # Convert AIMessage to string if needed
    if hasattr(answer_text, "content"):
        answer_text = answer_text.content

    matches = re.findall(r"\|([0-9,\s]+)\|", answer_text)
    ids = set()
    for match in matches:
        ids.update(id_.strip() for id_ in match.split(","))
    return list(ids)




# === Convert each sentence to a LangChain Document ===
documents = [
    Document(
        page_content=row["ref_excerpt"],
        metadata={
            "sentence_id": str(row["sentence_id"]),
            "case_id": str(row["case_id"]),
            "note_excerpt": row["note_excerpt"]
        }
    )
    for _, row in df.iterrows()
]


# === Prompt template via langchain ===

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a clinical assistant. Your goal is to answer the patient's question using only the sentences provided below.

- Every sentence used must be cited at the end using |sentence_id|.
- Cite all sentences that support each part of your answer.
- If multiple sentences support a point, cite all of them like |2,3|.
- Keep your total answer upto 75 words.
- Write one sentence per line.

Sentences:
{context}

Question:
{question}

Answer:
"""
)


# === Language model for answer generation ===

model_name = "gpt-4.1-mini" 

seed_val = 256

temp_val = 0.3


llm = ChatOpenAI(
    model_name=model_name, 
    temperature=temp_val,
    seed=seed_val,
    api_key= open_ai_api_key
)

qa_chain = prompt_template | llm



def ask_question_and_get_ids(query, case_id):
    
    # Just grab *all* sentences from the case, sorted by ID
    case_docs = sorted(
        [doc for doc in documents if doc.metadata["case_id"] == str(case_id)],
        key=lambda x: int(x.metadata["sentence_id"])
    )

    if not case_docs:
        return "", []

    context = "\n".join([f"{doc.metadata['sentence_id']}: {doc.page_content}" for doc in case_docs])

    # # Prepare the final prompt string (manually rendered)
    # prompt_text = prompt_template.format(context=context, question=query)

    # # Save prompt if a folder is provided
    # prompt_save_dir = 'prompts/'
    # if prompt_save_dir:
    #     os.makedirs(prompt_save_dir, exist_ok=True)
    #     prompt_path = os.path.join(prompt_save_dir, f"case_{case_id}.txt")
    #     with open(prompt_path, "w", encoding="utf-8") as f:
    #         f.write(prompt_text.strip())

    # Invoke the model
    answer = qa_chain.invoke({"context": context, "question": query})
    cited_ids = extract_cited_ids(answer)
    return answer, cited_ids



# === Create results folder if it doesn't exist ===
# results_folder = "results"
# os.makedirs(results_folder, exist_ok=True)

all_preds = []
all_labels = []

# For storing generated answers
results = []

for _, row in tqdm(cases.iterrows(), total=cases.shape[0], desc="Evaluating and Generating"):
    case_id = row["case_id"]
    question = row[question_column]

    answer, pred_ids = ask_question_and_get_ids(question, case_id)   #predicted
    answer = answer.content if hasattr(answer, "content") else str(answer)

    # Evaluate prediction vs gold labels
    case_df = df[df["case_id"] == case_id]

    gold_ids = case_df[case_df["relevance"] == "essential"]["sentence_id"].astype(str).tolist() #gold 

    # Store generation results
    results.append({
        "case_id": case_id,
        "question": question,
        "generated_answer": answer.strip(),
        "cited_sentence_ids": ", ".join(pred_ids),  
        "gold_essential_sentence_ids": ", ".join(gold_ids)
    })



    for _, sent in case_df.iterrows():
        gold = 1 if sent["binary_relevance"] == "essential" else 0
        pred = 1 if str(sent["sentence_id"]) in pred_ids else 0

        all_labels.append(gold)
        all_preds.append(pred)

# === Evaluation Metrics ===
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
cm = confusion_matrix(all_labels, all_preds)

print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1 Score: {f1:.3f}")

# # === Confusion Matrix Plot ===
# from sklearn.metrics import ConfusionMatrixDisplay
# import matplotlib.pyplot as plt
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Relevant", "Essential"])
# disp.plot(cmap=plt.cm.Blues)
# plt.title("Confusion Matrix - Sentence Classification")
# # plt.savefig(os.path.join(results_folder, "confusion_matrix.png"))
# plt.show()

# === Save as DataFrame ===
results_df = pd.DataFrame(results)
# import os
# results_df.to_excel(os.path.join(results_folder, "generated_answers_test.xlsx"), index=False)

# === Save in JSON submission format ===
submission_json = [
    {
        "case_id": str(row["case_id"]),
        "answer": row["generated_answer"]
    }
    for _, row in results_df.iterrows()
]

# import json
# with open(os.path.join(results_folder, "submission.json"), "w", encoding="utf-8") as f:
#     json.dump(submission_json, f, indent=4)