import argparse
import torch
import psutil
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict

def train_model(dataset_path, model_name, output_dir, num_train_epochs, batch_size):
    # Load dataset
    dataset = load_dataset('parquet', data_files=dataset_path)

    # Split the dataset into train and test sets
    dataset = dataset['train'].train_test_split(test_size=0.2)

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

    # Initialize Trainer with custom compute_loss function
    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            loss = torch.nn.functional.cross_entropy(logits, labels)
            return (loss, outputs) if return_outputs else loss

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
    )

    # Monitor memory usage
    def memory_monitor():
        process = psutil.Process()
        mem_info = process.memory_info()
        print(f"Memory usage: {mem_info.rss / 1024 ** 2:.2f} MB")

    # Train model with memory monitoring
    for epoch in range(num_train_epochs):
        memory_monitor()
        trainer.train()
        memory_monitor()

    # Save model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on the provided dataset.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset file.")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="Model name or path.")
    parser.add_argument("--output_dir", type=str, default="./model_output", help="Directory to save the trained model.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training and evaluation.")
    args = parser.parse_args()

    train_model(args.dataset, args.model_name, args.output_dir, args.num_train_epochs, args.batch_size)
