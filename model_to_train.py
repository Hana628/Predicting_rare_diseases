import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import joblib
import os
import pickle

# Important: Print versions for debugging
print(f"NumPy version: {np.__version__}")
print(f"scikit-learn version: {pd.__version__}")
print(f"pandas version: {pd.__version__}")
print(f"joblib version: {joblib.__version__}")

# Create output directory
os.makedirs('model_output', exist_ok=True)

"""
Keeping the exact same RareDiseaseModel class from the notebook
No modifications to ensure compatibility with the app
"""
class RareDiseaseModel:
    def __init__(self, input_dim, n_classes, is_binary=False):
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.is_binary = is_binary or n_classes == 2
        self.model = self._build_model()

    def _build_model(self):
        return MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size=32,
            learning_rate='adaptive',
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            verbose=True
        )
    
    def train(self, X_train, y_train, epochs=50, batch_size=32, early_stopping=True, patience=10, verbose=1):
        self.model.max_iter = epochs
        self.model.batch_size = batch_size
        self.model.early_stopping = early_stopping
        self.model.n_iter_no_change = patience
        self.model.verbose = verbose > 0
        self.model.fit(X_train, y_train)
        print("✅ Training complete.")
    
    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred)
        }
        return metrics
    
    def save(self, filepath):
        joblib.dump(self.model, filepath)
    
    def load(self, filepath):
        self.model = joblib.load(filepath)

# Functions to process data - same as notebook
def process_mygene2_data(file_path, min_samples_per_class=2):
    print(f"Loading data from {file_path}")
    with open(file_path, 'r') as f:
        records = [json.loads(line) for line in f if line.strip()]
    
    print(f"Loaded {len(records)} records")
    df = pd.DataFrame(records)
    df['positive_phenotypes'] = df['positive_phenotypes'].apply(lambda x: x if isinstance(x, list) else [])
    
    all_phenotypes = sorted(list(set(p for plist in df['positive_phenotypes'] for p in plist if p)))
    print(f"Found {len(all_phenotypes)} unique phenotypes")
    
    # Create feature matrix
    phenotype_features = pd.DataFrame(
        {p: df['positive_phenotypes'].apply(lambda x: int(p in x)) for p in all_phenotypes}
    )
    
    # Process categories
    df['category'] = df['orpha_category'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else 'Unknown')
    
    # Filter classes with enough samples
    counts = df['category'].value_counts()
    valid_classes = counts[counts >= min_samples_per_class].index
    print(f"Found {len(valid_classes)} valid disease categories")
    
    df = df[df['category'].isin(valid_classes)].reset_index(drop=True)
    phenotype_features = phenotype_features.loc[df.index].reset_index(drop=True)
    
    final_df = pd.concat([phenotype_features, df[['category']]], axis=1)
    return final_df

def split_and_preprocess_data(X, y, test_size=0.2, random_state=42, handle_imbalance=True):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    numeric_features = X.columns
    print(f"Preprocessing {len(numeric_features)} numeric features")
    
    # Create exactly the same preprocessor as in the notebook
    preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), numeric_features)])
    
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    if handle_imbalance:
        try:
            smote = SMOTE(random_state=random_state, k_neighbors=2)
            X_train_processed, y_train = smote.fit_resample(X_train_processed, y_train)
            print("✅ Applied SMOTE with k_neighbors=2")
        except Exception as e:
            print(f"⚠️ Warning: SMOTE failed with error {e}. Proceeding without SMOTE.")
            
    return X_train_processed, X_test_processed, y_train, y_test, preprocessor

# Main execution
def main():
    # Check paths
    mygene2_path = "C:\\Users\\hanar\\OneDrive\\Desktop\\rare_disease_app_clean\\model_output\\mygene2_5.7.22.txt"
    hp_terms_path = "C:\\Users\\hanar\\OneDrive\\Desktop\\rare_disease_app_clean\\model_output\\hp_terms.csv"
    
    print(f"Checking if input files exist:")
    print(f"  mygene2_path exists: {os.path.exists(mygene2_path)}")
    print(f"  hp_terms_path exists: {os.path.exists(hp_terms_path)}")
    
    # Process data
    print("\nStep 1: Processing data")
    df = process_mygene2_data(file_path=mygene2_path, min_samples_per_class=2)
    X = df.drop(columns=['category'])
    y = df['category']
    
    # Split and preprocess
    print("\nStep 2: Splitting and preprocessing data")
    X_train, X_test, y_train, y_test, preprocessor = split_and_preprocess_data(X, y)
    
    # Encode labels
    print("\nStep 3: Encoding labels")
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    # Train model
    print("\nStep 4: Training model")
    model = RareDiseaseModel(input_dim=X_train.shape[1], n_classes=len(label_encoder.classes_))
    model.train(X_train, y_train_encoded, epochs=100, batch_size=32, early_stopping=True, patience=10, verbose=1)
    
    # Evaluate
    print("\nStep 5: Evaluating model")
    metrics = model.evaluate(X_test, y_test_encoded)
    print("Evaluation Metrics:")
    print(metrics['classification_report'])
    
    # Save everything exactly as in the notebook
    print("\nStep 6: Saving model components")
    os.makedirs('model_output', exist_ok=True)
    
    # Important: Save the WHOLE RareDiseaseModel instance for compatibility
    print("Saving RareDiseaseModel instance...")
    joblib.dump(model, 'model_output/rare_disease_model_safe.joblib')
    
    print("Saving preprocessor...")
    joblib.dump(preprocessor, 'model_output/preprocessor_safe.joblib')
    
    print("Saving label encoder...")
    joblib.dump(label_encoder, 'model_output/label_encoder_safe.joblib')
    
    # Also save HP terms
    print("Copying HP terms")
    hp_df = pd.read_csv(hp_terms_path)
    hp_df.to_csv('model_output/hp_terms.csv', index=False)
    
    print("\n✅ All components saved successfully!")
    print("Model components saved in the model_output/ directory")

if __name__ == "__main__":
    main()