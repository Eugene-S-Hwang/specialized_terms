"""Code based on the code for "Leveraging Contextual Embeddings for Detecting Diachronic Semantic Shift" by Martinc et al."""

import os
import re
import argparse

import torch
import numpy as np
import nltk

import gc
from transformers import (
    AutoTokenizer, 
    AutoModel
)
from datasets import load_dataset

NUMBER_RE = re.compile(r"\d+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
PERIODS = [
        [2010, 2014],
        [2015, 2019],
        [2020, 2025]
    ]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--data_dir", type=str, required=True,
                    help="Folder of .txt files (searched recursively).")
    p.add_argument("--model", type=str, required=True,
                   help="Model to use")
    return p.parse_args()

def extract_year(filename: str):
    return filename[0:2]

def display_year(year: str) -> str:
    y = int(year)
    return f"19{year}" if y >= 93 else f"20{year:0>2}"

def load_data(folder, year_range):
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith(".txt"))

    for i, fname in enumerate(files):
        year_prefix = extract_year(fname)
        try:
            year = int(display_year(year_prefix))
        except ValueError:
            continue  # skip files with unrecognized year prefix

        if year < year_range[0] or year > year_range[1]:
            continue
        
        path = os.path.join(folder, fname)
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            content = fh.read()
            yield content

        if (i + 1) % 500 == 0:
            print(f"  processed {i + 1} files")


def tokens_to_batches(folder, year_range, tokenizer, batch_size, max_length):

    batches = []
    batch = []
    batch_counter = 0
    sent_tokenizer = nltk.sent_tokenize

    files = load_data(folder, year_range)
    total_batches = 0

    for text in files:

        text = NUMBER_RE.sub("", text)
        text = re.sub(r"\s+", " ", text)  # collapse extra spaces


        sents = []

        for sent in sent_tokenizer(text):
            sent = sent.strip().lower()

            marked_sent = "[CLS] " + sent + " [SEP]"
            sents.append(marked_sent)


        marked_text = " ".join(sents)

        tokenized_text = tokenizer.tokenize(marked_text)

        for i in range(0, len(tokenized_text), max_length):

            batch_counter += 1
            input_sequence = tokenized_text[i:i + max_length]

            indexed_tokens = tokenizer.convert_tokens_to_ids(input_sequence)

            batch.append((indexed_tokens, input_sequence))

            if batch_counter % batch_size == 0:
                batches.append(batch)
                total_batches += 1
                batch = []
        
        if(len(batches) > 1000):
            yield batches
            batches = []
    
    if len(batches) > 0:
        yield batches

    print()
    print('Tokenization done!')
    print('Number of batches: ', total_batches)

def get_token_embeddings(batches, model):

    token_embeddings = []
    tokenized_text = []
    counter = 0

    for batch in batches:
        batch_size = len(batch)
        counter += 1
        if counter % 1000 == 0:
            print('Generating embedding for batch: ', counter)
        lens = [len(x[0]) for x in batch]
        max_len = max(lens)
        tokens_tensor = torch.zeros(batch_size, max_len, dtype=torch.long).cuda()
        segments_tensors = torch.ones(batch_size, max_len, dtype=torch.long).cuda()
        batch_idx = [x[0] for x in batch]
        batch_tokens = [x[1] for x in batch]

        for i in range(batch_size):
            length = len(batch_idx[i])
            for j in range(max_len):
                if j < length:
                    tokens_tensor[i][j] = batch_idx[i][j]

        # Predict hidden states features for each layer
        with torch.no_grad():
            model_output = model(tokens_tensor, segments_tensors)
            encoded_layer = model_output[-1][-1:] #last layer of the encoder (concatenate the last four layers when using normal BERT model)


        for batch_i in range(batch_size):

            # For each token in the sentence...
            for token_i in range(len(batch_tokens[batch_i])):

                vec = encoded_layer[0][batch_i][token_i]

                token_vec = vec.detach().cpu().numpy().reshape(1, -1)

                token_embeddings.append(token_vec)
                tokenized_text.append(batch_tokens[batch_i][token_i])

    return token_embeddings, tokenized_text


def average_save_and_print(vocab_vectors, embeddings_path, age=None):
    for k, v in vocab_vectors.items():

        if len(v) == 2:
            avg = v[0] / v[1]
            vocab_vectors[k] = avg

    # Split dict into parallel arrays for .npz storage
    keys = list(vocab_vectors.keys())
    vectors = np.array([vocab_vectors[k].flatten() for k in keys])

    if age:
        save_path = f"{embeddings_path}/{age}.npz"
    else:
        save_path = f"{embeddings_path}/embeddings.npz"

    np.savez(save_path, keys=keys, vectors=vectors)

def get_time_embeddings(folder, embeddings_path, tokenizer, model, batch_size, max_length):
    for period in PERIODS:
        vocab_vectors = {}

        all_batches = tokens_to_batches(folder, period, tokenizer, batch_size, max_length)
        num_chunk = 0

        age = f"{period[0]}-{period[1]}"

        for batches in all_batches:
            num_chunk += 1
            print('Chunk ', num_chunk)

            token_embeddings, tokenized_text = get_token_embeddings(batches, model)

            splitted_tokens = []
            splitted_array = np.zeros((1, 768))
            prev_token = ""
            prev_array = np.zeros((1, 768))

            for i, token_i in enumerate(tokenized_text):

                array = token_embeddings[i]

                if token_i.startswith('##'):

                    if prev_token:
                        splitted_tokens.append(prev_token)
                        prev_token = ""
                        splitted_array = prev_array

                    splitted_tokens.append(token_i)
                    splitted_array += array

                else:

                    if token_i + '_' + age in vocab_vectors:
                        vocab_vectors[token_i + '_' + age][0] += array
                        vocab_vectors[token_i + '_' + age][1] += 1
                    else:
                        vocab_vectors[token_i + '_' + age] = [array, 1]

                    if splitted_tokens:
                        sarray = splitted_array / len(splitted_tokens)
                        stoken_i = "".join(splitted_tokens).replace('##', '')


                        if stoken_i + '_' + age in vocab_vectors:
                            vocab_vectors[stoken_i + '_' + age][0] += sarray
                            vocab_vectors[stoken_i + '_' + age][1] += 1
                        else:
                            vocab_vectors[stoken_i + '_' + age] = [sarray, 1]

                        splitted_tokens = []
                        splitted_array = np.zeros((1, 768))

                    prev_array = array
                    prev_token = token_i

            del token_embeddings
            del tokenized_text
            del batches
            gc.collect()

        print(f'Sentence embeddings for {age} generated.')

        print("Length of vocab after training: ", len(vocab_vectors.items()))

        average_save_and_print(vocab_vectors, embeddings_path, age)

### ----- WIKIPEDIA ---------------------------------------------------
def load_sentences(path: str, max_sentences=None) -> list[str]:
    sentences = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if len(line) > 20:
                sentences.append(line)
            if max_sentences and len(sentences) >= max_sentences:
                break
    return sentences
 
 
def sentences_to_batches(sentences, tokenizer, batch_size, max_length):
    """Same batching approach as your arXiv extraction script, minus the
    year/period filtering."""
    batches = []
    batch = []
    batch_counter = 0
    total_batches = 0
 
    for sent in sentences:
        sent = NUMBER_RE.sub("", sent)
        sent = re.sub(r"\s+", " ", sent).strip().lower()
        if not sent:
            continue
 
        marked_sent = "[CLS] " + sent + " [SEP]"
        tokenized_text = tokenizer.tokenize(marked_sent)
 
        for i in range(0, len(tokenized_text), max_length):
            batch_counter += 1
            input_sequence = tokenized_text[i:i + max_length]
            indexed_tokens = tokenizer.convert_tokens_to_ids(input_sequence)
            batch.append((indexed_tokens, input_sequence))
 
            if batch_counter % batch_size == 0:
                batches.append(batch)
                total_batches += 1
                batch = []
 
        if len(batches) > 1000:
            yield batches
            batches = []
 
    if batch:  # leftover partial batch
        batches.append(batch)
    if len(batches) > 0:
        yield batches
 
    print(f"Tokenization done! Number of batches: {total_batches}")
 
def get_wikipedia_embeddings(folder, embeddings_path, tokenizer, model, batch_size, max_length):
    vocab_vectors = {}

    sentences = load_sentences(folder)

    all_batches = sentences_to_batches(sentences, tokenizer, batch_size, max_length)
    num_chunk = 0

    for batches in all_batches:
        num_chunk += 1
        print("Chunk", num_chunk)
 
        token_embeddings, tokenized_text = get_token_embeddings(batches, model)
 
        splitted_tokens = []
        splitted_array = np.zeros((1, 768))
        prev_token = ""
        prev_array = np.zeros((1, 768))
 
        for i, token_i in enumerate(tokenized_text):
            array = token_embeddings[i]
 
            if token_i.startswith("##"):
                if prev_token:
                    splitted_tokens.append(prev_token)
                    prev_token = ""
                    splitted_array = prev_array
                splitted_tokens.append(token_i)
                splitted_array += array
            else:
                if token_i in vocab_vectors:
                    vocab_vectors[token_i][0] += array
                    vocab_vectors[token_i][1] += 1
                else:
                    vocab_vectors[token_i] = [array, 1]
 
                if splitted_tokens:
                    sarray = splitted_array / len(splitted_tokens)
                    stoken_i = "".join(splitted_tokens).replace("##", "")
 
                    if stoken_i in vocab_vectors:
                        vocab_vectors[stoken_i][0] += sarray
                        vocab_vectors[stoken_i][1] += 1
                    else:
                        vocab_vectors[stoken_i] = [sarray, 1]
 
                    splitted_tokens = []
                    splitted_array = np.zeros((1, 768))
 
                prev_array = array
                prev_token = token_i
 
        del token_embeddings
        del tokenized_text
        del batches
        gc.collect()
 
    print("General corpus embeddings generated.")
    print("Vocab size:", len(vocab_vectors))
 
    average_save_and_print(vocab_vectors, embeddings_path)

if __name__ == '__main__':
    args = parse_args()

    if args.model == "distilBERT-dapt":
        path = args.data_dir
        arxiv_category = '/'.join(path.split('/')[-2:])

        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)
        model = AutoModel.from_pretrained(f"models/distilBERT/{arxiv_category}", output_hidden_states=True)

        model.cuda()
        model.eval()

        embeddings_path = f"embeddings/{arxiv_category}/distilBERT-dapt"
        if not os.path.exists(embeddings_path):  
            os.makedirs(embeddings_path)

        get_time_embeddings(path, embeddings_path, tokenizer, model, args.batch_size, args.max_length)

    elif args.model == "distilBERT-base":
        path = args.data_dir
        arxiv_category = '/'.join(path.split('/')[-2:])

        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)
        model = AutoModel.from_pretrained('distilbert-base-uncased', output_hidden_states=True)

        model.cuda()
        model.eval()

        if path != "wiki_sentences.txt":
            embeddings_path = f"embeddings/{arxiv_category}/distilBERT-base"
            if not os.path.exists(embeddings_path):
                os.makedirs(embeddings_path)
            get_time_embeddings(path, embeddings_path, tokenizer, model, args.batch_size, args.max_length)
        else:
            embeddings_path = "embeddings/wikipedia/distilBERT-base"
            if not os.path.exists(embeddings_path):
                os.makedirs(embeddings_path)
            if not os.path.exists("wiki_sentences.txt"):
                wiki = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
                with open("wiki_sentences.txt", "w") as f:
                    count = 0
                    for article in wiki:
                        for chunk in article["text"].split("\n\n"):
                            chunk = chunk.strip()
                            if 20 < len(chunk) < 1000:
                                f.write(chunk + "\n")
                                count += 1
                        if count > 20000:
                            break
            get_wikipedia_embeddings(path, embeddings_path, tokenizer, model, args.batch_size, args.max_length)
            
            

        
