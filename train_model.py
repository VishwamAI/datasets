import argparse
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

def train_model(dataset_path, model_name, output_dir, num_train_epochs, batch_size):
    # Load dataset
    dataset = load_dataset('parquet', data_files=dataset_path)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer([q + " " + c for q, c in zip(examples['question'], examples['context'])], padding='max_length', truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir='./logs',
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
    )

    # Train model
    trainer.train()

    # Save model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on the provided dataset.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset file.")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="Model name or path.")
    parser.add_argument("--output_dir", type=str, default="./model_output", help="Directory to save the trained model.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training and evaluation.")
    args = parser.parse_args()

    train_model(args.dataset, args.model_name, args.output_dir, args.num_train_epochs, args.batch_size)
