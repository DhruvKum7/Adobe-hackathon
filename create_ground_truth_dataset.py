# create_ground_truth_dataset.py (Simplified Version)

import os
import json
import pandas as pd
from feature_extractor import extract_features

def build_dataset_from_ground_truth():
    # This is the only change: we now look for PDFs in the current folder '.'
    pdf_folder = '.'
    json_folder = '.' 
    
    all_labeled_data = []
    found_match = False

    print("Building dataset from ground truth files...")
    for json_filename in os.listdir(json_folder):
        if json_filename.lower().endswith('.json'):
            pdf_filename = os.path.splitext(json_filename)[0]
            if ' (' in pdf_filename:
                pdf_filename = pdf_filename.split(' (')[0] + '.pdf'
            else:
                pdf_filename = pdf_filename + '.pdf'

            pdf_path = os.path.join(pdf_folder, pdf_filename)

            if os.path.exists(pdf_path):
                found_match = True
                print(f"  -> SUCCESS: Matched '{json_filename}' with '{pdf_filename}'")
                
                with open(os.path.join(json_folder, json_filename), 'r', encoding='utf-8') as f:
                    ground_truth = json.load(f)
                
                truth_title = ground_truth.get('title', '')
                truth_headings = {item['text']: item['level'] for item in ground_truth.get('outline', [])}

                features_df = extract_features(pdf_path)
                if features_df.empty:
                    continue

                labels = []
                for index, row in features_df.iterrows():
                    text = row['text'].strip()
                    if text == truth_title.strip():
                        labels.append('Title')
                    elif text in truth_headings:
                        labels.append(truth_headings[text])
                    else:
                        labels.append('Other')
                
                features_df['label'] = labels
                all_labeled_data.append(features_df)

    if not found_match:
        print("\nCould not find any matching PDF/JSON pairs.")
        return

    final_df = pd.concat(all_labeled_data, ignore_index=True)
    final_df.to_csv('labeled_final.csv', index=False)
    
    print("\nSuccessfully created a new, accurate 'labeled_final.csv'.")

if __name__ == '__main__':
    build_dataset_from_ground_truth()