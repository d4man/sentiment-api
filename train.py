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
    # 1) Load dataset
    ds = load_dataset("tweet_eval", "sentiment")

    # 2) Tokenizer & model
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForSequenceClassification.from_pretrained(
    #     model_name, num_labels=3
    # )

    model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=3
    ).to(device)


    # 3) Tokenize (parallel)
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
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # 4) TrainingArguments (final)
    args = TrainingArguments(
        output_dir="./fine_tuned_tweet_eval",
        num_train_epochs=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        eval_steps=500,
        save_steps=500,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=2,

        # Load best model
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",

        # Speed & precision
        fp16=True,
        gradient_accumulation_steps=2,
        dataloader_num_workers=4,

        # Disable W&B prompts
        report_to=[],

        # Logging
        logging_dir="./logs",
        logging_steps=50,
    )

    # 5) Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tokenizer,
    )

    # 6) Train & Evaluate
    trainer.train()
    print("Final eval:", trainer.evaluate())

    # 7) Save
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Saved to", args.output_dir)

if __name__ == "__main__":
    main()
