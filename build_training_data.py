import os
import pandas as pd
from feature_extractor import extract_features 

# The folder where you have placed your training PDFs
PDF_FOLDER = 'input'
# The name for the output CSV file
OUTPUT_CSV = 'features_to_label.csv'

def create_master_feature_file():
    all_features = []
    
    if not os.path.exists(PDF_FOLDER):
        print(f"Error: The folder '{PDF_FOLDER}' was not found.")
        print("Please create this folder and add your PDF files to it.")
        return

    print(f"Scanning for PDFs in '{PDF_FOLDER}'...")
    for filename in os.listdir(PDF_FOLDER):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(PDF_FOLDER, filename)
            print(f"  -> Processing {filename}...")
            
            features_df = extract_features(pdf_path)
            features_df['source_file'] = filename 
            all_features.append(features_df)
    
    if not all_features:
        print(f"No PDFs were found in the {PDF_FOLDER} folder.")
        return

    # This line is now corrected
    master_df = pd.concat(all_features, ignore_index=True)
    master_df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\nSuccessfully created '{OUTPUT_CSV}'.")
    print("Next: Open this file, add a 'label' column, and save it as 'labeled_final.csv'.")

if __name__ == '__main__':
    create_master_feature_file()