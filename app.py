import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import pickle
import sys
from sklearn.neural_network import MLPClassifier

# Print Python and library versions for debugging
print(f"Python version: {sys.version}")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"Joblib version: {joblib.__version__}")

# Set page config
st.set_page_config(
    page_title="Rare Disease Prediction",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define paths
MODEL_DIR = "model_output"
MODEL_PATH = os.path.join(MODEL_DIR, "rare_disease_model_safe.joblib")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor_safe.joblib")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder_safe.joblib")
HP_TERMS_PATH = os.path.join(MODEL_DIR, "hp_terms.csv")

# Print file existence for debugging
print(f"MODEL_PATH exists: {os.path.exists(MODEL_PATH)}")
print(f"PREPROCESSOR_PATH exists: {os.path.exists(PREPROCESSOR_PATH)}")
print(f"LABEL_ENCODER_PATH exists: {os.path.exists(LABEL_ENCODER_PATH)}")
print(f"HP_TERMS_PATH exists: {os.path.exists(HP_TERMS_PATH)}")

# Define the RareDiseaseModel class needed for model loading
class RareDiseaseModel:
    def __init__(self, input_dim=None, n_classes=None, is_binary=False):
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.is_binary = is_binary or (n_classes == 2)
        self.model = None
        
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
    
    def predict(self, X):
        if self.model is not None:
            return self.model.predict(X)
        else:
            raise ValueError("Model not initialized")

# Custom safe loader for joblib files
def safe_load(file_path):
    """Safely load a joblib file with fallback methods"""
    print(f"Attempting to load: {file_path}")
    
    try:
        # Standard joblib loading
        return joblib.load(file_path)
    except Exception as e1:
        print(f"Standard joblib load failed: {str(e1)}")
        try:
            # Try pickle protocol
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e2:
            print(f"Pickle load failed: {str(e2)}")
            try:
                # Try reading bytes directly
                with open(file_path, 'rb') as f:
                    data = f.read()
                    print(f"Successfully read file as bytes, length: {len(data)}")
                    # But we can't do anything with raw bytes
                    return None
            except Exception as e3:
                print(f"Bytes reading failed: {str(e3)}")
                return None

# Load model and tools with enhanced error handling
def load_model():
    try:
        print("Starting model loading...")
        
        # Check if files exist
        if not os.path.exists(MODEL_PATH):
            print(f"Model file not found: {MODEL_PATH}")
            return None, None, None
            
        if not os.path.exists(PREPROCESSOR_PATH):
            print(f"Preprocessor file not found: {PREPROCESSOR_PATH}")
            return None, None, None
            
        if not os.path.exists(LABEL_ENCODER_PATH):
            print(f"Label encoder file not found: {LABEL_ENCODER_PATH}")
            return None, None, None
        
        # Try to load model
        model_data = safe_load(MODEL_PATH)
        if model_data is None:
            print("Failed to load model")
            return None, None, None
            
        # Determine model type and convert if needed
        if isinstance(model_data, RareDiseaseModel):
            print("Loaded RareDiseaseModel instance")
            model = model_data
        else:
            print("Loaded raw model, wrapping in RareDiseaseModel")
            wrapper = RareDiseaseModel()
            wrapper.model = model_data
            model = wrapper
            
        # Load preprocessor and label encoder
        preprocessor = safe_load(PREPROCESSOR_PATH)
        if preprocessor is None:
            print("Failed to load preprocessor")
            return None, None, None
            
        label_encoder = safe_load(LABEL_ENCODER_PATH)
        if label_encoder is None:
            print("Failed to load label encoder")
            return None, None, None
            
        print("All model components loaded successfully")
        return model, preprocessor, label_encoder
        
    except Exception as e:
        print(f"Unexpected error during model loading: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None

# Load phenotype mapping
@st.cache_data
def load_hp_terms():
    try:
        if not os.path.exists(HP_TERMS_PATH):
            print(f"HP terms file not found: {HP_TERMS_PATH}")
            return pd.DataFrame(), {}
            
        hp_df = pd.read_csv(HP_TERMS_PATH)
        hp_dict = dict(zip(hp_df['id'], hp_df['name']))
        print(f"Loaded {len(hp_df)} HP terms")
        return hp_df, hp_dict
    except Exception as e:
        print(f"Error loading HP terms: {str(e)}")
        return pd.DataFrame(), {}

# Try loading the model and data
try:
    model, preprocessor, label_encoder = load_model()
    hp_df, hp_terms_dict = load_hp_terms()
    model_loaded = model is not None and preprocessor is not None and label_encoder is not None
    print(f"Model loading complete. Success: {model_loaded}")
except Exception as e:
    print(f"Error during initialization: {str(e)}")
    import traceback
    traceback.print_exc()
    model_loaded = False
    model, preprocessor, label_encoder = None, None, None
    hp_df, hp_terms_dict = pd.DataFrame(), {}

# App Header
st.title("🧬 Rare Disease Prediction App")
st.markdown("Predict rare disease categories based on selected phenotypes (HPO terms).")

# Sidebar
st.sidebar.header("Select Phenotypes (HPO IDs)")
selected_phenos = []

if not hp_df.empty:
    selected_phenos = st.sidebar.multiselect(
        "Choose phenotypes",
        options=hp_df['id'].tolist(),
        format_func=lambda x: f"{x} - {hp_terms_dict.get(x, '')}"
    )
else:
    st.sidebar.warning("Phenotype mapping file (hp_terms.csv) not found!")

# Get all phenotypes from preprocessor
all_phenotypes = []
if model_loaded and preprocessor is not None:
    try:
        if hasattr(preprocessor, 'feature_names_in_'):
            all_phenotypes = preprocessor.feature_names_in_.tolist()
        elif hasattr(preprocessor, 'transformers_'):
            # For a ColumnTransformer object
            all_phenotypes = preprocessor.transformers_[0][1].feature_names_in_.tolist()
        else:
            # Try to get columns from a mock dataframe to see what it accepts
            all_phenos_set = set(hp_df['id'].tolist())
            input_df = pd.DataFrame({p: [1] for p in all_phenos_set})
            test_result = preprocessor.transform(input_df)
            all_phenotypes = input_df.columns.tolist()
            print(f"Inferred feature names: {all_phenotypes}")
    except Exception as e:
        print(f"Error getting phenotype list: {str(e)}")
        import traceback
        traceback.print_exc()
        all_phenotypes = []

# Prediction
if st.sidebar.button("Predict Disease Category"):
    if not selected_phenos:
        st.warning("Please select at least one phenotype.")
    elif not model_loaded:
        st.error("Model components could not be loaded. Please check the model files.")
    elif not all_phenotypes:
        st.error("Phenotype feature list could not be loaded from preprocessor!")
    else:
        try:
            # Build input vector
            if all_phenotypes:
                input_vector = np.array([int(pheno in selected_phenos) for pheno in all_phenotypes]).reshape(1, -1)
                input_df = pd.DataFrame(input_vector, columns=all_phenotypes)
            else:
                all_phenos_set = set(selected_phenos)
                input_df = pd.DataFrame({p: [1 if p in all_phenos_set else 0] for p in all_phenos_set})
            
            print(f"Input shape: {input_df.shape}")
            print(f"Input columns: {input_df.columns.tolist()}")
            
            # Preprocess input
            input_processed = preprocessor.transform(input_df)
            print(f"Processed input shape: {input_processed.shape}")
            
            # Predict
            if hasattr(model, 'predict'):
                pred_encoded = model.predict(input_processed)[0]
            else:
                pred_encoded = model.model.predict(input_processed)[0]
            
            print(f"Prediction (encoded): {pred_encoded}")
            
            pred_label = label_encoder.inverse_transform([pred_encoded])[0]
            print(f"Prediction (label): {pred_label}")
            
            # Display prediction result
            st.success(f"✅ **Predicted Disease Category:** {pred_label}")
            
            # Show selected phenotype names
            if hp_terms_dict:
                matched_names = [f"{p} — {hp_terms_dict.get(p, 'Unknown')}" for p in selected_phenos]
                st.markdown("### Selected Phenotypes:")
                for pheno_with_name in matched_names:
                    st.write(f"- {pheno_with_name}")
                    
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            print(f"Prediction error: {str(e)}")
            import traceback
            traceback.print_exc()
else:
    st.info("👈 Select phenotypes from sidebar and click 'Predict Disease Category'.")

# Display informative messages about the app's state
if not model_loaded:
    st.warning("""
    ### Model files not found or could not be loaded!
    Please ensure the following files exist in the `model_output` directory:
    - rare_disease_model_safe.joblib
    - preprocessor_safe.joblib
    - label_encoder_safe.joblib
    
    These files are created by running the training notebook.
    """)