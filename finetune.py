#!/usr/bin/env python3
"""Fine-tuning script for sentiment analysis models."""

import argparse
import json
import os
from typing import List, Dict, Any
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

def load_data(file_path: str) -> List[Dict[str, str]]:
    """Load training data from JSONL file."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def prepare_dataset(data: List[Dict[str, str]], tokenizer: Any) -> Dataset:
    """Prepare dataset for training."""
    label_map = {"negative": 0, "positive": 1}
    
    texts = [item['text'] for item in data]
    labels = [label_map[item['label']] for item in data]
    
    tokenized = tokenizer(texts, padding=True, truncation=True, max_length=512)
    
    return Dataset.from_dict({
        'input_ids': tokenized['input_ids'],
        'attention_mask': tokenized['attention_mask'],
        'labels': labels
    })

def train_model(data_file: str, epochs: int = 3, lr: float = 3e-5, output_dir: str = "./model") -> None:
    """Train sentiment analysis model."""
    data = load_data(data_file)
    
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    dataset = prepare_dataset(data, tokenizer)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        learning_rate=lr,
        logging_steps=10,
        save_strategy="epoch",
        overwrite_output_dir=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

def main() -> None:
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-data", "--data", required=True)
    parser.add_argument("-epochs", "--epochs", type=int, default=3)
    parser.add_argument("-lr", "--learning_rate", type=float, default=3e-5)
    parser.add_argument("-output", "--output_dir", default="./model")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train_model(args.data, args.epochs, args.learning_rate, args.output_dir)

if __name__ == "__main__":
    main()
