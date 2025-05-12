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
    #    Splits: train (~6.9K), validation (~1.8K), test (~2K)
    ds = load_dataset("tweet_eval", "sentiment")

    # 2) Choose our base model and tokenizer
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 3) Tokenize the texts (padding/truncation to max_length=128)
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=128,
        )
    ds = ds.map(tokenize, batched=True)

    # 4) Set the format for PyTorch
    ds.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"],
    )

    # 5) Load model with 3 labels (positive/negative/neutral)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
    )

    # 6) Configure training arguments
    output_dir = "./fine_tuned_tweet_eval"
    args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,

    # 1) Use half-precision (FP16)
    fp16=True,

    # # 2) Evaluate every epoch
    # evaluation_strategy="epoch",

    # 3) Speed up data loading
    dataloader_num_workers=4,

    # 4) Simulate a larger batch via accumulation
    gradient_accumulation_steps=2,

    logging_dir="./logs",
    logging_steps=50,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)


    # 7) Define a Trainer
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

    # 9) Save the fine-tuned model & tokenizer
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}/")

if __name__ == "__main__":
    main()
