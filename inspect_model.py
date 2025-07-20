# inspect_model.py

import pickle
from sklearn.tree import export_text

MODEL_PATH = 'heading_model.pkl'
ENCODER_PATH = 'label_encoder.pkl'
FEATURE_NAMES = [
    'font_size', 'is_bold', 'is_all_caps',
    'y_position', 'word_count', 'relative_size'
]

try:
    # Load the trained model and encoder
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)

    # Generate a text report showing the rules of the decision tree
    tree_rules = export_text(
        model, 
        feature_names=FEATURE_NAMES, 
        class_names=list(label_encoder.classes_)
    )

    print("--- Rules Learned by the Model ---")
    print(tree_rules)
    print("---------------------------------")
    print("\nThis shows the 'if/then' logic the model is using.")
    print("If it's very simple or only leads to 'Other', it means the training data was not distinct enough.")

except FileNotFoundError:
    print(f"Error: Could not find '{MODEL_PATH}' or '{ENCODER_PATH}'.")
    print("Please make sure you have successfully run 'train_model.py' first.")
except Exception as e:
    print(f"An error occurred: {e}")