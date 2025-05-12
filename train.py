# train.py

import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

def main():
    # 1) Load the tweet_eval sentiment dataset
    ds = load_dataset("tweet_eval", "sentiment")

    # 2) Base model and tokenizer
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 3) Tokenization
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=128,
        )
    ds = ds.map(tokenize, batched=True, num_proc=4, remove_columns=["text"])

    # 4) PyTorch format
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # 5) Load model with 3 labels
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
    )

    # 6) TrainingArguments (all in one place!)
    output_dir = "./fine_tuned_tweet_eval"
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=2,

        # batch sizes
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,

        # evaluation & saving per epoch so load_best_model works
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",

        # speedups & precision
        fp16=True,
        gradient_accumulation_steps=2,
        dataloader_num_workers=4,

        # logging
        logging_dir="./logs",
        logging_steps=50,
    )

    # 7) Create Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tokenizer,
    )

    # 8) Run training & evaluation
    trainer.train()
    metrics = trainer.evaluate()
    print("Eval metrics:", metrics)

    # 9) Save model & tokenizer
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model and tokenizer saved to {output_dir}/")

if __name__ == "__main__":
    main()
