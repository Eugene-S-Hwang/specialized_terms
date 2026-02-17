"""
Instructions to use Amazon S3 Buckets (if needed):
1) Access Amazon S3 Buckets
2) Download files from the buckets

NOTE: Must pay to download the data
"""


"""
Instructions:
1) Use Gcloud to download paper (free)
2) Use pypdf to get the text from the pdf file 
3) Use the arxiv API to get the category of the paper (date of paper can be identified through paper's id)
"""
import urllib.request as libreq
import feedparser
from pypdf import PdfReader
from google.cloud import storage
# from supabase import create_client, Client
# from dotenv import load_dotenv
import pandas as pd
import os


#### NOT USING SUPABASE CURRENTLY (CSV IS FASTER)
# load_dotenv()

# url = os.environ.get("SUPABASE_URL")
# key = os.environ.get("SUPABASE_KEY")

# supabase = create_client(url, key)

df = pd.read_csv("id_to_category.csv")
category_dict = dict(zip(df['paper_id'], df['category']))

def download_papers(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client(project="changeling-lab")
    bucket = storage_client.bucket(bucket_name, user_project="changeling-lab")

    blobs = bucket.list_blobs(prefix=source_blob_name, match_glob="**.pdf")

    test = 0
    for blob in blobs:
        if(test > 5):
            break
        test += 1

        blob_id = blob.name.split('/')[-1][:-6]
        category = find_category_csv(blob_id)

        # print(blob_id)

        if('.' in category):
            main_cat, sub_cat = category.split('.')
            dir_path = os.path.join(destination_file_name, f"{main_cat}/{sub_cat}")
        else:
            dir_path = os.path.join(destination_file_name, category)
        
        final_path = os.path.join(dir_path, f"{blob_id}.pdf")
        file_dir = os.path.dirname(final_path)

        os.makedirs(file_dir, exist_ok=True)

        blob.download_to_filename(final_path)

        print(f"Blob {blob.name} downloaded to {final_path}.")

## Slow?
def find_category_arXiv_API(id):

    url = f'http://export.arxiv.org/api/query?search_query=all:{id}&start=0&max_results=1'

    # print(url)
    with libreq.urlopen(url) as text:
        data = text.read()

    parse = feedparser.parse(data)

    print(parse.entries[0]["arxiv_primary_category"]["term"])

    # with libreq.urlopen('http://export.arxiv.org/api/query?search_query=all:neutron&start=0&max_results=1') as url:
    #   r = url.read()
    # print(r)

def find_category_csv(id):
    return category_dict.get(id)

def read_paper(file_name):
    reader = PdfReader(f"test_download_papers/{file_name}")

    for page in reader.pages:
        text = page.extract_text()
        print(text)

download_papers("arxiv-dataset", "arxiv/arxiv/pdf/2601/", "/data/user_data/ehwang2/papers")

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