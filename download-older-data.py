"""
Instructions:
1) Use Gcloud to download paper (free)
2) Use pymupdf to get the text from the pdf file 
3) Use the id_to_category.csv file to get the category of the paper (date of paper can be identified through paper's id)
"""
import urllib.request as libreq
import feedparser
from pypdf import PdfReader
from google.cloud import storage
import pandas as pd
import os
import fitz

df = pd.read_csv("id_to_category.csv", dtype={"paper_id":str, "category":str})
category_dict = dict(zip(df['paper_id'], df['category']))

new_subject_dict = {
    "acc-phys" : "physics.acc-ph",
    "adap-org" : "nlin.AO",
    "alg-geom" : "math.AG",
    "ao-sci" : "physics.ao-ph",
    "astro-ph" : "astro-ph",
    "atom-ph" : "physics.atom-ph",
    "bayes-an" : 'physics.data-an',
    "chao-dyn" : "nlin.CD",
    "chem-ph" : 'physics.chem-ph',
    "cmp-lg" : 'cs.CL',
    "comp-gas" : "nlin.CG",
    "dg-ga" : "math.DG",
    "funct-an" : "math.FA",
}

subject = "hep-ex"

def download_papers(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client(project="changeling-lab")
    bucket = storage_client.bucket(bucket_name, user_project="changeling-lab")

    blobs = bucket.list_blobs(prefix=source_blob_name, match_glob="**.pdf")

    blobs_dict = {}
    for blob in blobs:
        try:
            blob_name = blob.name.split('/')[-1] 
            blob_id = blob_name.split('v')[0]

            blobs_dict[blob_id] = blob
        except Exception as e:
            print(f"Skipping adding file due to error {e}")
    
    for (blob_id, blob) in blobs_dict.items():
        try:
            category = find_category_csv(blob_id)

            if('.' in category):
                main_cat, sub_cat = category.split('.')
                dir_path = os.path.join(destination_file_name, f"{main_cat}/{sub_cat}")
            else:
                dir_path = os.path.join(destination_file_name, category)
                
            final_path = os.path.join(dir_path, f"{blob_id}-{subject}.txt")
            file_dir = os.path.dirname(final_path)

            os.makedirs(file_dir, exist_ok=True)

            pdf_bytes = blob.download_as_bytes()

            doc = fitz.open(stream=pdf_bytes, filetype="pdf")

            text = ""
            for page in doc:
                text += page.get_text()  
            
            with open(final_path, "w", encoding="utf-8") as f:
                f.write(text)
            
            doc.close()

            print(f"Blob {blob.name} downloaded to {final_path}.")
        
        except Exception as e:
            print(f"Skipping downloading file due to error {e}")

def find_category_csv(id):
    if(subject not in new_subject_dict):
        return category_dict.get(f"{subject}/{id}")
    return new_subject_dict[subject]

download_papers("arxiv-dataset", f"arxiv/{subject}/pdf/", "/data/datasets/arXiv")