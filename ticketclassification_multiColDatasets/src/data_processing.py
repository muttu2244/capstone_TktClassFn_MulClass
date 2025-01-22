import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_data(input_path, output_path):
    # Load dataset
    data = pd.read_csv(input_path)

    # Handle datetime
    data['Date of Purchase'] = pd.to_datetime(data['Date of Purchase'])
    data['Purchase_Year'] = data['Date of Purchase'].dt.year
    data['Purchase_Month'] = data['Date of Purchase'].dt.month
    data['Purchase_Day'] = data['Date of Purchase'].dt.day
    data = data.drop(columns=['Date of Purchase'])

    # Encode categorical columns
    categorical_cols = ['Customer Gender', 'Product Purchased', 'Ticket Type', 'Ticket Channel', 'Ticket Priority']
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    # Vectorize text columns
    text_cols = ['Ticket Subject', 'Ticket Description']
    tfidf = TfidfVectorizer(max_features=500)

    for col in text_cols:
        tfidf_matrix = tfidf.fit_transform(data[col]).toarray()
        tfidf_df = pd.DataFrame(tfidf_matrix, columns=[f"{col}_tfidf_{i}" for i in range(tfidf_matrix.shape[1])])
        data = pd.concat([data, tfidf_df], axis=1)

    data = data.drop(columns=text_cols)

    # Save processed data
    data.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")

# Example usage
if __name__ == "__main__":
    preprocess_data("data/raw/customer_support_tickets.csv", "data/processed/preprocessed_tickets.csv")
