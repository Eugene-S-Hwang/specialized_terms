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
# from supabase import create_client, Client
# from dotenv import load_dotenv
import pandas as pd
import os
import fitz


#### NOT USING SUPABASE CURRENTLY (CSV IS FASTER)
# load_dotenv()

# url = os.environ.get("SUPABASE_URL")
# key = os.environ.get("SUPABASE_KEY")

# supabase = create_client(url, key)

#####

df = pd.read_csv("id_to_category.csv", dtype={"paper_id":str, "category":str})
category_dict = dict(zip(df['paper_id'], df['category']))

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
                
            final_path = os.path.join(dir_path, f"{blob_id}.txt")
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


## Slow?
def find_category_arXiv_API(id):

    url = f'http://export.arxiv.org/api/query?search_query=all:{id}&start=0&max_results=1'

    # print(url)
    with libreq.urlopen(url) as text:
        data = text.read()

    parse = feedparser.parse(data)

    print(parse.entries[0]["arxiv_primary_category"]["term"])


def find_category_csv(id):
    return category_dict.get(id)

def read_paper(file_name):
    reader = PdfReader(f"test_download_papers/{file_name}")

    for page in reader.pages:
        text = page.extract_text()
        print(text)

months = ['01', '02', '03', '04', '01', '06', '07', '08', '09', '10', '11', '12']
for m in months:
    download_papers("arxiv-dataset", f"arxiv/arxiv/pdf/17{m}/", "/data/datasets/arXiv")
# download_papers("arxiv-dataset", f'arxiv/arxiv/pdf/1701/', "test_download_papers")


# def find_category_supabase(id):
#     try:
#         response = (
#             supabase.table("CategoryFinder")
#             .select("category")
#             .eq("paper_id", id)
#             .execute()
#         )
        
#         return response.data[0]['category']
#     except:
#         print("SOME ISSUE")

# print("Testing")

# read_paper("2601.22155v1.pdf")