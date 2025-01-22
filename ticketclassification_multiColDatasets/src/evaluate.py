import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Paths
processed_test_data_path = "data/processed/preprocessed_tickets.csv"  # Replace with your actual test dataset path
model_path = "models/model.pkl"

# Load the Processed Test Data
print("Loading test data...")
df = pd.read_csv(processed_test_data_path)

# Define Feature Categories
categorical_features = [
    "Customer Gender", "Product Purchased", "Ticket Type",
    "Ticket Status", "Resolution", "Ticket Priority"
]
numerical_features = ["Customer Age"]
target_column = "Ticket Priority"  # Modify if the target column differs

# Dynamically identify TF-IDF columns for 'Ticket Description'
tfidf_columns = [col for col in df.columns if col.startswith("Ticket Description_tfidf_")]

# Combine all features
features = tfidf_columns + categorical_features + numerical_features

# Check for missing columns
for col in features + [target_column]:
    if col not in df.columns:
        raise ValueError(f"Missing column in test dataset: {col}")

# Prepare Test Data
X_test = df[features]
y_test = df[target_column]

# One-hot encode categorical features
X_test = pd.get_dummies(X_test, columns=categorical_features, drop_first=True)

# Align columns with training data (handle missing or extra dummy columns)
print("Aligning test data with model features...")
model = joblib.load(model_path)
model_features = model.feature_importances_.shape[0]
X_test = X_test.reindex(columns=model_features, fill_value=0)

# Load Model
print("Loading model...")
try:
    model = joblib.load(model_path)
    print("Model loaded successfully.")
except FileNotFoundError:
    raise FileNotFoundError(f"Model file not found at {model_path}. Please ensure the model is trained and saved.")

# Evaluate Model
print("Evaluating model...")
y_pred = model.predict(X_test)

# Print Evaluation Metrics
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))
