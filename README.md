# LIMICS@BioNLP-2025
ArchEHR-QA: BioNLP at ACL 2025 Shared Task on Grounded Electronic Health Record Question Answering

This repository contains the code developed by the LIMICS team for the BioNLP 2025 Shared Task. It was designed to reproduce the experiments and results presented in our paper. The implementation includes all necessary components to prompt the LLMs, train the models, preprocess the data and postprocess the data.

## Zero-Shot Prompting using GPT 4.1-mini

Run the zero-shot LLM generation and evaluation pipeline:

```bash
poetry run python archehr/prompting/zeroShot.py
