import os
import json
import pandas as pd
import pickle
from feature_extractor import extract_features 

def predict_structure(pdf_path, model, label_encoder):
    """
    Predicts the document structure of a single PDF file.
    """
    print("  [Debug] Extracting features...")
    features_df = extract_features(pdf_path)
    
    if features_df.empty:
        print(f"  [Debug] Could not extract features from {pdf_path}. Skipping.")
        return None
    print(f"  [Debug] Extracted {len(features_df)} text blocks.")

    feature_columns = [
        'font_size', 'is_bold', 'is_all_caps',
        'y_position', 'word_count', 'relative_size'
    ]
    for col in feature_columns:
        if col not in features_df.columns:
            features_df[col] = 0
            
    X_predict = features_df[feature_columns]

    print("  [Debug] Making predictions with the model...")
    predictions_encoded = model.predict(X_predict)
    predictions = label_encoder.inverse_transform(predictions_encoded)
    print("  [Debug] Predictions are complete.")
    
    doc_title = "Untitled Document"
    outline = []
    
    title_candidates = features_df.loc[predictions == 'Title']
    if not title_candidates.empty:
        doc_title = title_candidates.sort_values(by='y_position').iloc[0]['text']

    for i, row in features_df.iterrows():
        label = predictions[i]
        if label in ['H1', 'H2', 'H3']:
            outline.append({
                "level": label,
                "text": row['text'],
                "page": int(row['page_num'])
            })
    
    print("  [Debug] Assembling final JSON structure.")
    return {
        "title": doc_title,
        "outline": outline
    }

if __name__ == '__main__':
    INPUT_DIR = '/app/input'
    OUTPUT_DIR = '/app/output'
    MODEL_PATH = 'heading_model.pkl'
    ENCODER_PATH = 'label_encoder.pkl'

    if not os.path.exists('input'): os.makedirs('input')
    if not os.path.exists('output'): os.makedirs('output')
    
    process_input_dir = INPUT_DIR if os.path.exists(INPUT_DIR) else 'input'
    process_output_dir = OUTPUT_DIR if os.path.exists(OUTPUT_DIR) else 'output'

    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(ENCODER_PATH, 'rb') as f:
            label_encoder = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Model files not found. Please make sure '{MODEL_PATH}' and '{ENCODER_PATH}' are in your project directory.")
        exit()

    print(f"Starting processing for PDFs in '{process_input_dir}'...")
    for filename in os.listdir(process_input_dir):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(process_input_dir, filename)
            print(f"Processing: {filename}")
            
            try:
                result = predict_structure(pdf_path, model, label_encoder)
                
                if result:
                    output_filename = os.path.splitext(filename)[0] + ".json"
                    output_path = os.path.join(process_output_dir, output_filename)
                    
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(result, f, indent=4)
                    
                    print(f"  -> Successfully created {output_filename}")
                else:
                    print(f"  -> No result was generated for {filename}.")
            
            except Exception as e:
                print(f"  [ERROR] An error occurred while processing {filename}: {e}")
    
    print("Processing complete.")