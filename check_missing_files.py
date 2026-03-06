import os
from google.cloud import storage

# Setup
client = storage.Client(project="changeling-lab")
bucket = client.bucket("arxiv-dataset", user_project="changeling-lab")
local_root = "/data/datasets/arXiv/"
prefix = "arxiv/arxiv/pdf/2601/"

print("Checking for missing files...")
# blobs = bucket.list_blobs(prefix=prefix, match_glob="**.pdf")

missing_count = 0
for i in range(14017, 23287):
    # filename = os.path.basename(blob.name)[:-6] + '.pdf'
    filename = '2601.' + '0' * (5 - len(str(i))) + str(i) + '.pdf'
    # This checks if the file exists ANYWHERE in your categorized subfolders
    # You may need to adjust the logic if you want to check specific subdirs
    file_exists = any(filename in files for _, _, files in os.walk(local_root))
    
    if not file_exists:
        print(f"MISSING: {filename}")
        missing_count += 1

print(f"Audit complete. Total missing: {missing_count}")