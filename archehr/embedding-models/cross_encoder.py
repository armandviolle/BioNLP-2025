from archehr.utils import load_data
import pandas as pd
from archehr import BASE_DIR
from sklearn.model_selection import train_test_split
from datasets import Dataset
from sentence_transformers import CrossEncoder
import torch
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
from sentence_transformers.evaluation.SentenceEvaluator import SentenceEvaluator
from sentence_transformers.cross_encoder import (
    CrossEncoderTrainingArguments,
    CrossEncoderTrainer,
)
from pathlib import Path
import numpy as np
from scipy import stats


class BinaryEvaluator(SentenceEvaluator):
    def __init__(self, sentence_pairs, labels, batch_size, name="Validation"):
        self.sentence_pairs = sentence_pairs
        self.labels = labels
        self.batch_size = batch_size
        self.name = name

    def __call__(self, model, output_path=None, epoch=-1, steps=-1):
        # Process in batches with progress bar
        pred_labels = []
        for i in tqdm(
            range(0, len(self.sentence_pairs), self.batch_size),
            desc="Processing Validation Data",
        ):
            batch_pairs = self.sentence_pairs[i : i + self.batch_size]
            prediction = model.predict(batch_pairs)
            pred_labels.extend(list(np.where(prediction < 0.5, 0, 1)))

        # Metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.labels, pred_labels, zero_division=0, average="binary", pos_label=1
        )
        print(f"F1-Score: {f1:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")

        return {"f1": f1, "precision": precision, "recall": recall}


def finetune_on_augmented_dataset(output_dir: Path, logging_dir: Path):
    """
    Fine-tune the CrossEncoder model on the augmented dataset.
    Args:
        output_dir (Path): Directory to save the model.
        logging_dir (Path): Directory to save the logs.
    """
    # Load the data
    pretrain_data = pd.read_csv(BASE_DIR / "data" / "augmented_dataset.csv")
    pretrain_data = pretrain_data[
        ["question_generated", "ref_excerpt", "binary_relevance"]
    ].rename(
        columns={
            "question_generated": "query",
            "ref_excerpt": "answer",
            "binary_relevance": "label",
        }
    )

    # Set random seed for reproducibility
    np.random.seed(42)
    # Split into train (15) and validation (5)
    pretrain_data, preval_data = train_test_split(
        pretrain_data, test_size=0.25, shuffle=True
    )
    # Convert to PyTorch Dataset
    processed_pretrain_data = Dataset.from_dict(pretrain_data)
    processed_preval_data = Dataset.from_dict(preval_data)

    # Load the pre-trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L12-v2",
        num_labels=1,
        device=device,
        activation_fn=torch.nn.Sigmoid(),
    )
    # Customize for binary classification
    label2id = {"essential": 1}
    model.model.config.label2id = label2id
    model.model.config.id2label = {v: k for k, v in model.model.config.label2id.items()}

    # Define the loss function
    # Use Binary Cross-Entropy Loss
    loss = BinaryCrossEntropyLoss(model)

    # Set the evaluator
    # Use the BinaryEvaluator for binary classification
    pairs = list(zip(processed_preval_data["query"], processed_preval_data["answer"]))
    labels = processed_preval_data["label"]
    cls_evaluator = BinaryEvaluator(
        sentence_pairs=pairs,
        labels=labels,
        name="Validation",
        batch_size=128,
    )

    args = CrossEncoderTrainingArguments(
        output_dir=output_dir,
        logging_dir=logging_dir,
        logging_strategy="epoch",
        report_to="tensorboard",
        num_train_epochs=10,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        bf16=True,  # Set to True if you have a GPU that supports BF16
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        metric_for_best_model="f1",
        load_best_model_at_end=True,
        greater_is_better=True,
        eval_on_start=True,
        seed=42,
    )
    # Initialize the trainer
    trainer = CrossEncoderTrainer(
        model=model,
        args=args,
        train_dataset=processed_pretrain_data,
        eval_dataset=processed_preval_data,
        loss=loss,
        evaluator=cls_evaluator,
    )
    trainer.train()

    return model


def evaluate_scratch_model():
    """
    Evaluate the model without fine-tuning on the dev set.
    """
    # Load the data
    data = load_data(BASE_DIR / "data" / "dev")

    # Set random seed for reproducibility
    seeds = [i for i in range(100)]
    np.random.shuffle(seeds)
    f1s = []
    precisions = []
    recalls = []
    for seed in seeds:
        np.random.seed(seed)

        # Split into train (15) and validation (5)
        train_data, test_data = train_test_split(data, test_size=5, shuffle=True)

        def create_qa_pairs(example):
            question = example["narrative"].strip()
            pairs = []
            for sentence in example["sentences"]:
                # Check if any of the answer texts appears in this sentence
                label = int(sentence[2] == "essential")
                pairs.append({
                    "query": question,
                    "answer": sentence[1],
                    "label": label,  # Convert boolean to 1 or 0
                })

            return pairs

        # Process your dataset
        processed_test_data = []
        for example in tqdm(test_data, desc="Processing examples"):
            processed_test_data.extend(create_qa_pairs(example))

        processed_test_dataset = Dataset.from_list(processed_test_data)

        # Load model without fine-tuning
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_scratch = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L12-v2", num_labels=1, device=device
        )
        label2id = {"essential": 1}
        model_scratch.model.config.label2id = label2id
        model_scratch.model.config.id2label = {
            v: k for k, v in model_scratch.model.config.label2id.items()
        }

        pairs = list(
            zip(processed_test_dataset["query"], processed_test_dataset["answer"])
        )
        true_labels = processed_test_dataset["label"]
        pred_labels = np.where(model_scratch.predict(pairs) < 0, 0, 1)

        # Metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, pred_labels, zero_division=0, average="binary", pos_label=1
        )
        f1s.append(f1)
        precisions.append(precision)
        recalls.append(recall)
        print(f"F1-Score: {f1:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")

    # Calculate mean and 95% confidence interval
    f1_mean = np.mean(f1s)
    precision_mean = np.mean(precisions)
    recall_mean = np.mean(recalls)
    n = len(f1s)
    ci_f1 = stats.t.interval(
        0.95, df=n - 1, loc=f1_mean, scale=stats.sem(f1s)
    )  # 95% CI
    ci_precision = stats.t.interval(
        0.95, df=n - 1, loc=precision_mean, scale=stats.sem(precisions)
    )  # 95% CI
    ci_recall = stats.t.interval(
        0.95, df=n - 1, loc=recall_mean, scale=stats.sem(recalls)
    )  # 95% CI

    print(f"F1: {f1_mean:.4f} ({ci_f1[0]:.4f}, {ci_f1[1]:.4f})")
    print(
        f"Precision: {precision_mean:.4f} ({ci_precision[0]:.4f}, {ci_precision[1]:.4f})"
    )
    print(f"Recall: {recall_mean:.4f} ({ci_recall[0]:.4f}, {ci_recall[1]:.4f})")


def evaluate_model_finetuned_augmented(
    model_augmented, output_dir: Path, logging_dir: Path
):
    """
    Evaluate the model after fine-tuning on the augmented set.
    """
    # Load the data
    data = load_data(BASE_DIR / "data" / "dev")

    # Set random seed for reproducibility
    seeds = [i for i in range(100)]
    np.random.shuffle(seeds)
    f1s = []
    precisions = []
    recalls = []
    for seed in seeds:
        np.random.seed(seed)

        # Split into train (15) and validation (5)
        train_data, test_data = train_test_split(data, test_size=5, shuffle=True)

        def create_qa_pairs(example):
            question = example["narrative"].strip()
            pairs = []
            for sentence in example["sentences"]:
                # Check if any of the answer texts appears in this sentence
                label = int(sentence[2] == "essential")
                pairs.append({
                    "query": question,
                    "answer": sentence[1],
                    "label": label,  # Convert boolean to 1 or 0
                })

            return pairs

        # Process your dataset
        processed_test_data = []
        for example in tqdm(test_data, desc="Processing examples"):
            processed_test_data.extend(create_qa_pairs(example))

        processed_test_dataset = Dataset.from_list(processed_test_data)

        # Pairs for evaluation
        pairs = list(
            zip(processed_test_dataset["query"], processed_test_dataset["answer"])
        )
        true_labels = processed_test_dataset["label"]
        pred_labels = np.where(model_augmented.predict(pairs) < 0, 0, 1)

        # Metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, pred_labels, zero_division=0, average="binary", pos_label=1
        )
        f1s.append(f1)
        precisions.append(precision)
        recalls.append(recall)
        print(f"F1-Score: {f1:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")

    # Calculate mean and 95% confidence interval
    f1_mean = np.mean(f1s)
    precision_mean = np.mean(precisions)
    recall_mean = np.mean(recalls)
    n = len(f1s)
    ci_f1 = stats.t.interval(
        0.95, df=n - 1, loc=f1_mean, scale=stats.sem(f1s)
    )  # 95% CI
    ci_precision = stats.t.interval(
        0.95, df=n - 1, loc=precision_mean, scale=stats.sem(precisions)
    )  # 95% CI
    ci_recall = stats.t.interval(
        0.95, df=n - 1, loc=recall_mean, scale=stats.sem(recalls)
    )  # 95% CI

    print(f"F1: {f1_mean:.4f} ({ci_f1[0]:.4f}, {ci_f1[1]:.4f})")
    print(
        f"Precision: {precision_mean:.4f} ({ci_precision[0]:.4f}, {ci_precision[1]:.4f})"
    )
    print(f"Recall: {recall_mean:.4f} ({ci_recall[0]:.4f}, {ci_recall[1]:.4f})")


def evaluate_model_finetuned(output_dir: Path, logging_dir: Path):
    """
    Evaluate the model after fine-tuning on the dev set.
    """
    # Load the data
    data = load_data(BASE_DIR / "data" / "dev")

    # Set random seed for reproducibility
    seeds = [i for i in range(100)]
    np.random.shuffle(seeds)
    f1s = []
    precisions = []
    recalls = []
    for seed in seeds:
        np.random.seed(seed)

        # Split into train (15) and validation (5)
        train_data, test_data = train_test_split(data, test_size=5, shuffle=True)

        def create_qa_pairs(example):
            question = example["narrative"].strip()
            pairs = []
            for sentence in example["sentences"]:
                # Check if any of the answer texts appears in this sentence
                label = int(sentence[2] == "essential")
                pairs.append({
                    "query": question,
                    "answer": sentence[1],
                    "label": label,  # Convert boolean to 1 or 0
                })

            return pairs

        # Process your dataset
        processed_train_data = []
        for example in tqdm(train_data, desc="Processing examples"):
            processed_train_data.extend(create_qa_pairs(example))
        processed_test_data = []
        for example in tqdm(test_data, desc="Processing examples"):
            processed_test_data.extend(create_qa_pairs(example))

        processed_train_dataset = Dataset.from_list(processed_train_data)
        processed_test_dataset = Dataset.from_list(processed_test_data)

        # Load model scratch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        final_model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L12-v2",
            num_labels=1,
            device=device,
            activation_fn=torch.nn.Sigmoid(),
        )
        # Customize for medical domain (optional but recommended)
        label2id = {"essential": 1}
        final_model.model.config.label2id = label2id
        final_model.model.config.id2label = {
            v: k for k, v in final_model.model.config.label2id.items()
        }

        loss = BinaryCrossEntropyLoss(final_model)

        args = CrossEncoderTrainingArguments(
            output_dir=output_dir / f"seed_{seed}",
            logging_dir=logging_dir / f"seed_{seed}",
            logging_strategy="epoch",
            report_to="tensorboard",
            num_train_epochs=35,
            per_device_train_batch_size=64,
            per_device_eval_batch_size=64,
            learning_rate=2e-5,
            warmup_ratio=0.1,
            bf16=True,  # Set to True if you have a GPU that supports BF16
            save_strategy="epoch",
            save_total_limit=2,
            seed=42,
        )

        trainer = CrossEncoderTrainer(
            model=final_model,
            args=args,
            train_dataset=processed_train_dataset,
            loss=loss,
        )
        trainer.train()

        pairs = list(
            zip(processed_test_dataset["query"], processed_test_dataset["answer"])
        )
        true_labels = processed_test_dataset["label"]
        pred_labels = np.where(final_model.predict(pairs) < 0.5, 0, 1)

        # Metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, pred_labels, zero_division=0, average="binary", pos_label=1
        )
        f1s.append(f1)
        precisions.append(precision)
        recalls.append(recall)
        print(f"F1-Score: {f1:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")

    # Calculate mean and 95% confidence interval
    f1_mean = np.mean(f1s)
    precision_mean = np.mean(precisions)
    recall_mean = np.mean(recalls)
    n = len(f1s)
    ci_f1 = stats.t.interval(
        0.95, df=n - 1, loc=f1_mean, scale=stats.sem(f1s)
    )  # 95% CI
    ci_precision = stats.t.interval(
        0.95, df=n - 1, loc=precision_mean, scale=stats.sem(precisions)
    )  # 95% CI
    ci_recall = stats.t.interval(
        0.95, df=n - 1, loc=recall_mean, scale=stats.sem(recalls)
    )  # 95% CI

    print(f"F1: {f1_mean:.4f} ({ci_f1[0]:.4f}, {ci_f1[1]:.4f})")
    print(
        f"Precision: {precision_mean:.4f} ({ci_precision[0]:.4f}, {ci_precision[1]:.4f})"
    )
    print(f"Recall: {recall_mean:.4f} ({ci_recall[0]:.4f}, {ci_recall[1]:.4f})")


if __name__ == "__main__":
    # Evaluate scratch model
    evaluate_scratch_model()

    # Evaluate model finetuned on dev set
    output_dir = BASE_DIR / "data" / "models" / "cross_encoder_model_finetuned"
    logging_dir = BASE_DIR / "data" / "logs" / "cross_encoder_model_finetuned"
    # Create the directories if they don't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    logging_dir.mkdir(parents=True, exist_ok=True)

    # Run evaluation
    evaluate_model_finetuned(output_dir, logging_dir)

    # Evaluate model finetuned on augmented dataset
    # Define the output and logging directories
    output_dir = (
        BASE_DIR / "data" / "models" / "cross_encoder_model_augmented_finetuned"
    )
    logging_dir = BASE_DIR / "data" / "logs" / "cross_encoder_model_augmented_finetuned"
    # Create the directories if they don't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    logging_dir.mkdir(parents=True, exist_ok=True)

    # Fine-tune the model
    model_augmented = finetune_on_augmented_dataset(output_dir, logging_dir)

    # Evaluate the model
    evaluate_model_finetuned_augmented(model_augmented, output_dir, logging_dir)
