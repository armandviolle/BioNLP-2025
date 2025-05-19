import textwrap
import re
import pandas as pd
from tqdm import tqdm
from question_generation import generate_questions_summary
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
# import os

# -----------------------------------------------

# Load your OpenAI API Key

api_key = "YOUR_API_KEY"

# -----------------------------------------------
# MAIN SCRIPT

# Download the 2012 i2b2 challenge dataset '2012-07-15.original-annotation.release/'
# I2B2-2012 data requires signing the n2nb2 license.

df = generate_questions_summary(
    open_ai_key=api_key,
    i2b2_dataset_path="i2b2_data_dir",
    emrqa_dataset_hf_path="Eladio/emrqa-msquad",
    mimic_3_dataset_hf_path="Medilora/mimic_iii_diagnosis_anonymous",
)

qa_results = []
case_counter = 1  # Start case ID from 1

for i in tqdm(range(len(df))):
    context = df["note_excerpt"].iloc[i]
    query = df["generated_question"].iloc[i]
    data_source = df["source"].iloc[i]

    # --- Split into all sentences ---
    sentences = re.split(r"(?<=[.!?])\s+", context.strip())
    all_sentences = [s.strip() for s in sentences if s.strip()]

    # --- Embed all sentences and build retriever ---
    documents = [Document(page_content=sentence) for sentence in all_sentences]
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = FAISS.from_documents(documents, embeddings)

    # --- Get all sentences with similarity scores ---
    all_results = vectorstore.similarity_search_with_score(query, k=len(documents))

    # Top-k as essential
    top_k = 10
    essential_docs = all_results[:top_k]
    essential_sentences = [doc[0].page_content.strip() for doc in essential_docs]

    # Supplementary: take next 4 (ranks 8–10)
    supplementary_docs = all_results[top_k : top_k + 4]
    supplementary_sentences = [
        doc[0].page_content.strip() for doc in supplementary_docs
    ]

    # Not relevant : remaining ones
    used = set(essential_sentences + supplementary_sentences)
    not_relevant_sentences = [s for s in all_sentences if s not in used]

    for sentence_id, sentence in enumerate(all_sentences, start=1):
        if sentence in essential_sentences:
            label = "essential"
        elif sentence in supplementary_sentences:
            label = "supplementary"
        else:
            label = "not-relevant"

        qa_results.append({
            "case_id": case_counter,
            "note_excerpt": context,  # use original untrimmed note
            "question_generated": query,
            "sentence_id": sentence_id,
            "ref_excerpt": sentence,
            "relevance": label,
            "source": data_source,
        })

    case_counter += 1  # Increment case ID

# -----------------------------------------------
# Create DataFrame and save it

qa_results_df = pd.DataFrame(qa_results)

# Reorder columns
qa_results_df = qa_results_df[
    [
        "case_id",
        "note_excerpt",
        "question_generated",
        "sentence_id",
        "ref_excerpt",
        "relevance",
        "source",
    ]
]


# # Save to Excel

# print(qa_results_df)

# save_dir = "qa_generation/"
# os.makedirs(save_dir, exist_ok=True)

# qa_results_df.to_excel(os.path.join(save_dir, "qa_results_structured.xlsx"), index=False)
