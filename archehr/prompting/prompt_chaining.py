from archehr.utils import load_data
from archehr import BASE_DIR
from typing import List
import json
import os
from openai import OpenAI


def get_input(data, case_id):
    """
    This function formats the input for the OpenAI API.
    """

    patient_narrative = data[case_id]["narrative"].strip()
    clinical_note = data[case_id]["clinical_note"].strip()
    template_input = f"## Patient's narrative\n{patient_narrative}\n\n## Clinical note\n{clinical_note}"
    return template_input


def get_sentences(data, case_id):
    """
    This function formats the input for the OpenAI API.
    """
    output_answer = """"""
    for sent in data[case_id]["sentences"]:
        template_input = f"## Sentence id:{sent[0] + 1}\n{sent[1]}\n\n"
        output_answer += template_input
    return output_answer


def get_clinician_question(data, case_id):
    """
    This function formats the input for the OpenAI API.
    """
    patient_narrative = data[case_id]["narrative"].strip()
    clinician_question = data[case_id]["clinician_question"].strip()
    template_input = f"## Patient's narrative\n{patient_narrative}\n\n## Question reformulated by a clinician\n{clinician_question}"
    return template_input


def get_answer_size(answer_text):
    """
    Check if the answer is longer than 75 words.
    """
    answer_sentences = []
    for line in answer_text.split("\n"):
        """
        e.g., The final negative retrograde cholangiogram result after the procedure confirms that ERCP was an effective treatment for the patient's condition. |8|
        """
        if not line.strip():
            continue
        line_parts = line.rsplit("|", maxsplit=2)
        if len(line_parts) < 3:
            # No citations found
            sent = line
        else:
            sent = line_parts[-3]
        sent = sent.strip()

        # Add a period at the end of the sentence if it doesn't have one
        if sent and sent[-1] not in ".!?":
            sent += "."

        answer_sentences.append(sent)

    case_answer = " ".join(answer_sentences)
    case_answer_words = [w for w in case_answer.split(" ") if w.strip()]
    return len(case_answer_words)


def process_cases(seeds: List[int], output_folder: str):
    """
    This function processes the cases using the OpenAI API with Prompt Chaining method.
    """
    # Load the data
    data = load_data(BASE_DIR / "data" / "dev")

    # Load the OpenAI API key
    # You can set the OpenAI API key in your environment variables or directly in the code
    # os.environ["OPENAI_API_KEY"] = "your_openai_api_key"

    client = OpenAI(
        # This is the default and can be omitted
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    # Load system prompts
    system_prompt_path = BASE_DIR / "config" / "prompt_chaining"
    with open(system_prompt_path / "1-free_answer.txt", "r") as f:
        system_prompt_free_answer = f.read()
    with open(
        system_prompt_path / "2-essential_sentences_identification.txt", "r"
    ) as f:
        system_prompt_essential_sentences = f.read()
    with open(system_prompt_path / "3-answer_reformulation.txt", "r") as f:
        system_prompt_answer_reformulation = f.read()
    with open(system_prompt_path / "4-answer_compression.txt", "r") as f:
        system_prompt_answer_compression = f.read()
    with open(system_prompt_path / "5-strict_answer_compression.txt", "r") as f:
        system_prompt_strict_answer_compression = f.read()

    results = []
    max_retries = 3
    for seed_init in seeds:
        seed = seed_init
        for case_id in data.keys():
            finished = False
            while not finished:
                print("=" * 50)
                print(f"Processing case {case_id}...")
                print("=" * 50)
                answer = {"case_id": case_id}
                messages = [
                    {"role": "system", "content": system_prompt_free_answer},
                    {"role": "user", "content": get_input(data, case_id)},
                ]
                response = client.chat.completions.create(
                    model="o4-mini-2025-04-16",
                    messages=messages,
                    seed=seed,
                    frequency_penalty=0,
                    presence_penalty=0,
                )
                # Print the response
                print("Response:")
                print(response.choices[0].message.role)
                print(response.choices[0].message.content)

                messages.append({
                    "role": "assistant",
                    "content": response.choices[0].message.content,
                })
                messages.append({
                    "role": "system",
                    "content": system_prompt_essential_sentences,
                })
                messages.append({
                    "role": "user",
                    "content": get_sentences(data, case_id),
                })
                response = client.chat.completions.create(
                    model="gpt-4.1-mini-2025-04-14",
                    messages=messages,
                    temperature=0.3,
                    seed=seed,
                    frequency_penalty=0,
                    presence_penalty=0,
                )
                # Print the response
                print("Response:")
                print(response.choices[0].message.role)
                print(response.choices[0].message.content)

                messages.append({
                    "role": "assistant",
                    "content": response.choices[0].message.content,
                })
                messages.append({
                    "role": "system",
                    "content": system_prompt_answer_reformulation,
                })
                messages.append({
                    "role": "user",
                    "content": get_clinician_question(data, case_id),
                })
                response = client.chat.completions.create(
                    model="gpt-4.1-mini-2025-04-14",
                    messages=messages,
                    temperature=0.3,
                    seed=seed,
                    frequency_penalty=0,
                    presence_penalty=0,
                )

                print("Response:")
                print(response.choices[0].message.role)
                print(response.choices[0].message.content)
                print(get_answer_size(response.choices[0].message.content))

                if get_answer_size(response.choices[0].message.content) > 75:
                    messages.append({
                        "role": "assistant",
                        "content": response.choices[0].message.content,
                    })
                    messages.append({
                        "role": "user",
                        "content": system_prompt_answer_compression,
                    })
                    response = client.chat.completions.create(
                        model="gpt-4.1-mini-2025-04-14",
                        messages=messages,
                        temperature=0.3,
                        seed=seed,
                        frequency_penalty=0,
                        presence_penalty=0,
                    )
                    # Print the response
                    print("Response:")
                    print(response.choices[0].message.role)
                    print(response.choices[0].message.content)
                    print(get_answer_size(response.choices[0].message.content))

                for i in range(max_retries):
                    if get_answer_size(response.choices[0].message.content) > 75:
                        messages.append({
                            "role": "assistant",
                            "content": response.choices[0].message.content,
                        })
                        messages.append({
                            "role": "user",
                            "content": system_prompt_strict_answer_compression,
                        })
                        response = client.chat.completions.create(
                            model="gpt-4.1-mini-2025-04-14",
                            messages=messages,
                            temperature=0.3,
                            seed=seed,
                            frequency_penalty=0,
                            presence_penalty=0,
                        )
                        # Print the response
                        print("Response:")
                        print(response.choices[0].message.role)
                        print(response.choices[0].message.content)
                        print(get_answer_size(response.choices[0].message.content))
                    else:
                        break

                if get_answer_size(response.choices[0].message.content) <= 75:
                    answer["answer"] = response.choices[0].message.content
                    results.append(answer)
                    finished = True
                else:
                    seed += 1
                    print(f"Retrying with a new seed: {seed}")

        # Save the results in json format
        with open(output_folder / f"answer_{seed_init}.json", "w") as f:
            json.dump(results, f, indent=4)


if __name__ == "__main__":
    # Define the seeds
    seeds = [range(0, 1000, 100)]
    # Define the output folder
    output_folder = BASE_DIR / "data" / "results" / "prompt_chaining"
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    # Process the cases
    process_cases(seeds, output_folder)
