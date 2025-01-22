import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Step 1: Load the Model
model_pipeline = joblib.load("ticket_classification_model.pkl")  # Replace with the correct path
print("Model loaded successfully.")

# Step 2: Load New Data for Prediction
new_data_path = "new_tickets.csv"  # Replace with the file containing new data
df = pd.read_csv(new_data_path)

# Step 3: Data Processing
text_features = ['Ticket Description', 'Ticket Subject']
categorical_features = ['Ticket Priority', 'Ticket Type', 'Ticket Channel']
numerical_features = ['Customer Age', 'First Response Time', 'Time to Resolution']

# Ensure all required columns exist
required_columns = text_features + categorical_features + numerical_features
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Missing column in new dataset: {col}")

X_new = df[text_features + categorical_features + numerical_features]

# Apply preprocessing steps to match the training pipeline
text_transformer = Pipeline(steps=[("tfidf", joblib.load("tfidf_vectorizer.pkl"))])
categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])
numerical_transformer = Pipeline(steps=[("scaler", StandardScaler())])

preprocessor = ColumnTransformer(
    transformers=[
        ("text", text_transformer, text_features),
        ("cat", categorical_transformer, categorical_features),
        ("num", numerical_transformer, numerical_features),
    ]
)

# Transform X_new
X_new_transformed = preprocessor.fit_transform(X_new)

# Step 4: Predict
predictions = model_pipeline.predict(X_new_transformed)

# Step 5: Output Predictions
df['Predicted Ticket Status'] = predictions
output_path = "predicted_tickets.csv"
df.to_csv(output_path, index=False)
print(f"Predictions saved to {output_path}")
