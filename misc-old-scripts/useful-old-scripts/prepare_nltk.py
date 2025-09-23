# prepare_nltk.py
# This is a one-time utility script to be run on an internet-enabled node (e.g., a login node).
# Its purpose is to download all necessary NLTK data packages into a local directory,
# making them available for offline use by our experiment scripts.

import nltk
import os

# Define the target directory for the NLTK data.
DOWNLOAD_DIR = './nltk_data'
# Define all required NLTK packages.
PACKAGES = ['punkt', 'punkt_tab']

print(f"--- NLTK Offline Data Preparation ---")
print(f"Target download directory: {DOWNLOAD_DIR}")

# Create the directory if it doesn't exist.
if not os.path.exists(DOWNLOAD_DIR):
    os.makedirs(DOWNLOAD_DIR)
    print(f"Created directory: {DOWNLOAD_DIR}")

# Download all required packages to our specified local directory.
try:
    for package in PACKAGES:
        print(f"Downloading '{package}' package...")
        nltk.download(package, download_dir=DOWNLOAD_DIR)
    
    print("\n--- Success! ---")
    print("All required NLTK packages have been downloaded successfully.")
    print(f"Your project now contains the necessary offline data in the '{DOWNLOAD_DIR}' folder.")
    print("You can now run experiments that use nltk.sent_tokenize on the compute nodes.")

except Exception as e:
    print("\n--- ERROR ---")
    print(f"An error occurred during download: {e}")
    print("Please check your internet connection and permissions.")