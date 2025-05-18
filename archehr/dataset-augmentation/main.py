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
# Function 1 - Retrieve top-k relevant sentences
def retrieve_top_sentences(context, question, top_k=6):
    sentences = re.split(r'(?<=[.!?])\s+', context.strip())
    documents = [Document(page_content=sentence) for sentence in sentences if sentence.strip()]

    embeddings = OpenAIEmbeddings(openai_api_key = api_key)
    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    relevant_sentences = retriever.get_relevant_documents(question)
    
    return relevant_sentences

# -----------------------------------------------
# Function 2 - Wrap text nicely for printing (optional)
def wrap_text_preserve_newlines(text, width=110):
    lines = text.split('\n')
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    wrapped_text = '\n'.join(wrapped_lines)
    return wrapped_text

# -----------------------------------------------
# Function 3 - Process LLM response and print it (optional)
def process_llm_response(llm_response):
    print("\n" + "="*100)
    print("Response:")
    print(wrap_text_preserve_newlines(llm_response['result']))
    
    print("\nSources used:")
    sources_info = []
    for i, source in enumerate(llm_response["source_documents"], 1):
        print(f"\n--- Source {i} ---")
        source_info = source.metadata.get('source', 'Manual Text Input')
        print(f"Source Info: {source_info}")
        print(f"Page Content:\n{wrap_text_preserve_newlines(source.page_content)}")
        
        sources_info.append({
            "source_info": source_info,
            "page_content": source.page_content
        })
    
    print("="*100 + "\n")
    
    return {
        "response": llm_response['result'],
        "sources": sources_info
    }


# Function 4 - Sentence chunker (optional if needed separately)
def split_text_into_sentence_chunks(text, chunk_size=1000, chunk_overlap=200):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            if chunk_overlap > 0 and len(current_chunk) > chunk_overlap:
                overlap_text = current_chunk[-chunk_overlap:]
                current_chunk = overlap_text + " " + sentence
            else:
                current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return [Document(page_content=chunk) for chunk in chunks]

# -----------------------------------------------
# MAIN SCRIPT

# Download the 2012 i2b2 challenge dataset '2012-07-15.original-annotation.release/'
# I2B2-2012 data requires signing the n2nb2 license.

df = generate_questions_summary(open_ai_key = api_key ,i2b2_dataset_path = "i2b2_data_dir", emrqa_dataset_hf_path = "Eladio/emrqa-msquad" ,mimic_3_dataset_hf_path = "Medilora/mimic_iii_diagnosis_anonymous")

qa_results = []
case_counter = 1  # Start case ID from 1

for i in tqdm(range(len(df))):
    context = df['note_excerpt'].iloc[i]
    query = df['generated_question'].iloc[i]
    data_source = df['source'].iloc[i]

    # --- Split into all sentences ---
    sentences = re.split(r'(?<=[.!?])\s+', context.strip())
    all_sentences = [s.strip() for s in sentences if s.strip()]

    # --- Embed all sentences and build retriever ---
    documents = [Document(page_content=sentence) for sentence in all_sentences]
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 7})


    # --- Get all sentences with similarity scores ---
    all_results = vectorstore.similarity_search_with_score(query, k=len(documents))

    # Top-k as essential
    top_k = 10
    essential_docs = all_results[:top_k]
    essential_sentences = [doc[0].page_content.strip() for doc in essential_docs]

    # Supplementary: take next 3 (ranks 8–10)
    supplementary_docs = all_results[top_k:top_k+4]
    supplementary_sentences = [doc[0].page_content.strip() for doc in supplementary_docs]

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
            "source": data_source
        })



    case_counter += 1  # Increment case ID

# -----------------------------------------------
# Create DataFrame and save it

qa_results_df = pd.DataFrame(qa_results)

# Reorder columns
qa_results_df = qa_results_df[["case_id", "note_excerpt", "question_generated", "sentence_id", "ref_excerpt", "relevance", "source"]]


# # Save to Excel

# print(qa_results_df)

# save_dir = "qa_generation/"
# os.makedirs(save_dir, exist_ok=True)

# qa_results_df.to_excel(os.path.join(save_dir, "qa_results_structured.xlsx"), index=False)