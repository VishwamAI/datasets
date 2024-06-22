import argparse
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset, load_metric

def evaluate_model(dataset_path, model_path, batch_size):
    # Load dataset
    dataset = load_dataset('parquet', data_files=dataset_path)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Load metric
    metric = load_metric("accuracy")

    # Define evaluation function
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = torch.argmax(logits, dim=-1)
        return metric.compute(predictions=predictions, references=labels)

    # Evaluate model
    trainer = Trainer(
        model=model,
        eval_dataset=tokenized_datasets['test'],
        compute_metrics=compute_metrics,
    )

    # Get evaluation results
    results = trainer.evaluate()
    print(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model on the provided dataset.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset file.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model.")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for evaluation.")
    args = parser.parse_args()

    evaluate_model(args.dataset, args.model_path, args.batch_size)
