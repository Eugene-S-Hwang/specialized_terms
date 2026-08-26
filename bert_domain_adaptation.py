"""Code based on the code for "Tracing the Development of the Virtual Particle Concept Using Semantic Change Detection" and Claude"""

import os
import re
import argparse

import torch
from datasets import Dataset

from transformers import (
    AutoTokenizer, 
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True,
                    help="Folder of .txt files (searched recursively).")
    p.add_argument("--max_length", type=int, default=128,
                    help="Max sequence length. 128 keeps training fast; "
                         "raise to 256/512 only if your sentences are long "
                         "and you have GPU headroom.")
    p.add_argument("--mlm_probability", type=float, default=0.15)
    p.add_argument("--epochs", type=float, default=1.0,
                    help="1 epoch is usually enough for domain adaptation "
                         "(as opposed to pretraining from scratch).")
    p.add_argument("--batch_size", type=int, default=64,
                    help="DistilBERT is tiny, so you can push batch size much "
                         "higher than you would for BERT-base. Lower this "
                         "if you hit OOM on your GPU.")
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--save_steps", type=int, default=2000)
    p.add_argument("--logging_steps", type=int, default=100)
    p.add_argument("--max_files", type=int, default=None,
                    help="Optional cap on number of files read, for a quick "
                         "smoke test before committing to the full corpus.")
    p.add_argument("--eval_split", type=float, default=0.05,
                    help="Fraction of chunks held out for eval. Not used to "
                         "tune hyperparameters — just a sanity check that "
                         "MLM loss is decreasing and not overfitting. Set "
                         "to 0 to skip eval entirely (saves a bit of time).")
    p.add_argument("--fp16", action="store_true", default=None,
                    help="Mixed precision — safe speed win on any modern "
                         "NVIDIA GPU (T4/V100/A100/etc).")
    return p.parse_args()
 

def extract_year(filename: str):
    return filename[0:2]

def display_year(year: str) -> str:
    y = int(year)
    return f"19{year}" if y >= 93 else f"20{year:0>2}"

# Choose and load model
def load_model(max_seq_length):
    model = AutoModelForMaskedLM.from_pretrained('distilbert-base-uncased', output_hidden_states = True)
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased', use_fast=True, do_lower_case=True, max_len=max_seq_length)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print("loaded model:", model.name_or_path)
    print("device:", model.device)
    model.eval()
    return model, tokenizer

#Get chunks for text
def chunk_text(text: str) -> list[str]:
    NUMBER_RE = re.compile(r"\d+")
    SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

    text_no_numbers = NUMBER_RE.sub("", text)
    text_no_numbers = re.sub(r"\s+", " ", text_no_numbers)  # collapse extra spaces
    sentences = SENTENCE_SPLIT_RE.split(text_no_numbers)
    return [s.strip() for s in sentences if s.strip()]

# loads in all articles
# returns as list of sentences
def load_data(folder, max_files=None):
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith(".txt"))
    size = len(files)
    
    processed = 0

    for i, fname in enumerate(files):
        year_prefix = extract_year(fname)
        try:
            year = int(display_year(year_prefix))
        except ValueError:
            continue  # skip files with unrecognized year prefix

        if year < 2010:
            continue

        path = os.path.join(folder, fname)
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            content = fh.read()

        for chunk in chunk_text(content):
            if len(chunk) > 20:
                yield {"text": chunk}

        if (i + 1) % 500 == 0:
            print(f"  processed {i + 1} files")
        processed += 1

        if max_files and processed >= max_files:
            break

def create_dataset(chunk_size, train_split, test_split, tokenizer, args):

    # Dataset erstellen

    dataset = Dataset.from_generator(
        load_data,
        gen_kwargs={"folder": args.data_dir, "max_files": args.max_files})

    # tokenizer for dataset-library
    def tokenize_function(dataset):
        return tokenizer(dataset["text"], max_length=chunk_size, truncation=True, padding=False)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # create chunks with chunk_size
    # Drop last chunk that is < chunk_size
    def chunk_texts(examples):
    # Concatenate all texts
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        # Compute length of concatenated texts
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the last chunk if it's smaller than chunk_size
        total_length = (total_length // chunk_size) * chunk_size
        # Split by chunks of max_len
        result = {
            k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
            for k, t in concatenated_examples.items()
        }
        # Create a new labels column
        result["labels"] = result["input_ids"].copy()
        return result

    # In Chunks der Größe chunksize aufteilen
    chunk_size = chunk_size
    chunked_dataset = tokenized_dataset.map(chunk_texts, batched=True)

    # Train und Testdaten festlegen
    train_size = int(len(chunked_dataset) * train_split)
    test_size = int(len(chunked_dataset) * test_split)
    downsampled_dataset = chunked_dataset.train_test_split(train_size=train_size, test_size=test_size)

    return downsampled_dataset["train"], downsampled_dataset["test"]

def main():
    args = parse_args()
 
    print(f"Loading tokenizer/model from DistilBERT.")
    model, tokenizer = load_model(args.max_length)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_probability,
    )
 
    train_dataset, eval_dataset = create_dataset(chunk_size=128, train_split=0.8, test_split=0.2, tokenizer=tokenizer, args=args)

    folder_name = '/'.join(args.data_dir.split('/')[-2:])

    output_dir = f'models/distilBERT/{folder_name}'

    if not os.path.exists(output_dir):  
        os.makedirs(output_dir)
 
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_steps=args.save_steps,
        save_total_limit=2,
        logging_steps=args.logging_steps,
        fp16=args.fp16 if args.fp16 is not None else torch.cuda.is_available(),
        report_to="none",  # avoid wandb/etc prompts on a compute node
        dataloader_num_workers=4,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.save_steps,
        per_device_eval_batch_size=args.batch_size,
    )
 
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
 
    print("Starting training...")
    trainer.train()
 
    if eval_dataset is not None:
        metrics = trainer.evaluate()
        print(f"Final eval loss: {metrics['eval_loss']:.4f}")
 
    print(f"Saving final model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
 
    print("Done.")
 
 
if __name__ == "__main__":
    main()