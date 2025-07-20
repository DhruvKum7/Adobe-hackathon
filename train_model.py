import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pickle

# 1. Load the Labeled Data
try:
    df = pd.read_csv('labeled_final.csv')
except FileNotFoundError:
    print("Error: 'labeled_final.csv' not found. Please create this file from 'features_to_label.csv' first.")
    exit()

# 2. Prepare Data for Training
features = [
    'font_size', 'is_bold', 'is_all_caps',
    'y_position', 'word_count', 'relative_size'
]
X = df[features]
y = df['label']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# 4. Initialize and Train the Decision Tree Model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate the Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model trained with an accuracy of: {accuracy * 100:.2f}%")

# 6. Save the Trained Model and the Label Encoder
with open('heading_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print("Model and label encoder have been saved to 'heading_model.pkl' and 'label_encoder.pkl'")