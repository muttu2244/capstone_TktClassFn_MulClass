import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Paths
processed_data_path = "data/processed/processed_data.csv"
model_save_path = "models/model.pkl"

# Load Processed Data
df = pd.read_csv(processed_data_path)

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
        raise ValueError(f"Missing column in dataset: {col}")

# Prepare Data
X = df[features]
y = df[target_column]

# One-hot encode categorical features
X = pd.get_dummies(X, columns=categorical_features, drop_first=True)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
print("Training model...")
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save Model
print("Saving model...")
joblib.dump(model, model_save_path)

# Evaluate Model
print("Evaluating model...")
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))
