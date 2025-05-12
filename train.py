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

    # 2) Base model & tokenizer
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 3) Tokenize (parallel & cached)
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=128,
        )
    ds = ds.map(
        tokenize,
        batched=True,
        num_proc=4,
        remove_columns=["text"],
        load_from_cache_file=True,
    )

    # 4) PyTorch format
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # 5) Load model for 3-way classification
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
    )

    # 6) TrainingArguments using step-based eval & save
    output_dir = "./fine_tuned_tweet_eval"
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=2,

        # batch sizes
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,

        # step-based evaluation & checkpointing
        eval_steps=500,
        save_steps=500,
        save_total_limit=2,               # keep only last 2 checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",

        # speed & precision
        fp16=True,
        gradient_accumulation_steps=2,
        dataloader_num_workers=4,
        # avoid wandb prompt
        report_to=[],

        # logging
        logging_dir="./logs",
        logging_steps=50,
    )

    # 7) Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tokenizer,
    )

    # 8) Train & evaluate
    trainer.train()
    metrics = trainer.evaluate()
    print("Eval metrics:", metrics)

    # 9) Save final model & tokenizer
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model and tokenizer saved to {output_dir}/")

if __name__ == "__main__":
    main()
