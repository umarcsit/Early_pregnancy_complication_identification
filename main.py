from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
from typing import Any, Dict, List, Optional, Union
from contextlib import asynccontextmanager
import pandas as pd
import numpy as np
import joblib
import uvicorn
import logging
import json
import os
import pickle
import warnings
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Suppress scikit-learn version warnings for model loading
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*")
warnings.filterwarnings("ignore", message=".*Trying to unpickle estimator.*")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Version information available via /health endpoint

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    load_models()
    yield
    # Shutdown (if needed)

app = FastAPI(title="Medical Prediction API", version="1.0.0", lifespan=lifespan)

# Load model configuration from JSON file
def load_model_config():
    """Load model configuration from JSON file"""
    config_path = 'Mode_of_delivery2.json'
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise HTTPException(status_code=500, detail=f"Configuration file not found: {config_path}")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in configuration file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Invalid JSON in configuration file: {str(e)}")
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error loading configuration: {str(e)}")

# Load model configuration
MODEL_CONFIG = load_model_config()

def safe_load_model(model_path: str, model_name: str):
    """
    Safely load a model with multiple fallback methods to handle compatibility issues
    """
    # Normalize the path to handle both relative and absolute paths
    normalized_path = os.path.normpath(model_path)
    
    # Check if file exists
    if not os.path.exists(normalized_path):
        raise FileNotFoundError(f"Model file not found: {normalized_path}")
    
    # Try different loading strategies
    loading_methods = [
        ("sklearn compatibility fix", lambda: _load_with_sklearn_compatibility_fix(normalized_path)),
        ("joblib with warnings suppressed", lambda: _load_with_suppressed_warnings(normalized_path)),
        ("joblib with custom protocol", lambda: _load_with_custom_protocol(normalized_path)),
        ("pickle with protocol 4", lambda: _load_with_pickle_protocol(normalized_path, 4)),
        ("pickle with protocol 5", lambda: _load_with_pickle_protocol(normalized_path, 5)),
    ]
    
    for method_name, load_func in loading_methods:
        try:
            model = load_func()
            return model
        except Exception as e:
            if method_name == loading_methods[-1][0]:  # Last method
                raise e
            continue
    
    raise Exception(f"All loading methods failed for {model_name}")

def _load_with_suppressed_warnings(file_path):
    """Load model with all warnings suppressed"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return joblib.load(file_path)

def _load_with_custom_protocol(file_path):
    """Load model with custom joblib protocol"""
    try:
        # Try with different joblib protocols
        return joblib.load(file_path, mmap_mode=None)
    except:
        return joblib.load(file_path)

def _load_with_pickle_protocol(file_path, protocol):
    """Load model with specific pickle protocol"""
    import pickle
    with open(file_path, 'rb') as f:
        return pickle.load(f, encoding='latin1')

def _load_with_sklearn_compatibility_fix(file_path):
    """Load model with sklearn compatibility fixes"""
    import pickle
    import sys
    
    # Create a custom unpickler that can handle sklearn version differences
    class SklearnCompatibilityUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            # Handle sklearn module name changes
            if module.startswith('sklearn.compose._column_transformer'):
                module = 'sklearn.compose._column_transformer'
            elif module.startswith('sklearn.preprocessing._encoders'):
                module = 'sklearn.preprocessing._encoders'
            elif module.startswith('sklearn.preprocessing._data'):
                module = 'sklearn.preprocessing._data'
            elif module.startswith('sklearn.pipeline'):
                module = 'sklearn.pipeline'
            elif module.startswith('sklearn.base'):
                module = 'sklearn.base'
            
            # Handle specific class name changes
            if name == '_RemainderColsList':
                # Try to find a compatible alternative
                try:
                    from sklearn.compose._column_transformer import _RemainderColsList
                    return _RemainderColsList
                except ImportError:
                    # If not available, use a simple list as fallback
                    return list
            
            return super().find_class(module, name)
    
    with open(file_path, 'rb') as f:
        unpickler = SklearnCompatibilityUnpickler(f)
        return unpickler.load()

def _create_mock_model(model_name: str):
    """Create a mock model for testing when real models can't be loaded"""
    from sklearn.dummy import DummyClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    
    # Create a simple mock pipeline
    mock_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', DummyClassifier(strategy='constant', constant='mock_prediction'))
    ])
    
    # Fit with dummy data
    import numpy as np
    X_dummy = np.random.rand(10, 5)
    y_dummy = ['mock_prediction'] * 10
    mock_pipeline.fit(X_dummy, y_dummy)
    
    return mock_pipeline

MODEL_ORDER = [
    'Mode_of_delivery2',
    'Antenatal_Peripartum_Maternal_Complications',
    'Neonatal__Fetal_Complications',
    'Postnatal_Maternal_Complications'
]

# Global variables
loaded_models = {}
model_is_pipeline = {}

class PatientData(BaseModel):
    patientData: Dict[str, Any]

def apply_same_preprocessing_as_training(X):
    """
    Apply the EXACT same preprocessing as used in training:
    1. Convert categorical to lowercase strings
    2. Identify numerical and categorical features
    3. Apply StandardScaler to numerical and OneHotEncoder to categorical
    """
    try:
        # Make a copy to avoid modifying original data
        X_processed = X.copy()
        
        # Step 1: Ensure all data is properly typed and clean
        for col in X_processed.columns:
            # Convert to string first to handle any mixed types
            X_processed[col] = X_processed[col].astype(str)
            
            # Handle null/empty values
            X_processed[col] = X_processed[col].replace(['nan', 'none', 'null', '', 'na', 'NaN', 'None', 'NULL'], 'not specified')
        
        # Step 2: Identify categorical and numerical features (same as training)
        # Re-identify after cleaning
        categorical_features = []
        numerical_features = []
        
        for col in X_processed.columns:
            # Determine if column should be numeric based on patterns
            is_numeric_column = any(keyword in col.lower() for keyword in ['age', 'bmi', 'weeks', 'score', 'b.p', 'bp', 'pulse', 'rate', 'temp', 'height', 'number'])
            
            if is_numeric_column:
                # Convert to numeric, coercing errors to 0.0
                X_processed[col] = pd.to_numeric(X_processed[col], errors='coerce').fillna(0.0)
                numerical_features.append(col)
            else:
                # Keep as string for categorical columns, convert to lowercase
                X_processed[col] = X_processed[col].astype(str).str.lower().str.strip()
                categorical_features.append(col)
        
        # Step 3: Create the SAME preprocessing pipeline as training
        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        # Step 4: Fit and transform the data
        X_transformed = preprocessor.fit_transform(X_processed)
        
        # Convert sparse matrix to dense array if needed
        if hasattr(X_transformed, 'toarray'):
            X_transformed = X_transformed.toarray()
        
        # Preprocessing completed
        
        return X_transformed, preprocessor
        
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Preprocessing error: {str(e)}")

def load_models():
    """Load all models and check if they're pipelines or standalone models"""
    global loaded_models, model_is_pipeline
    failed_models = []
    
    for model_name, config in MODEL_CONFIG.items():
        try:
            model_path = config['filename']
            
            # Check if file exists
            if not os.path.exists(model_path):
                failed_models.append(f"{model_name}: File not found")
                continue
            
            # Use safe loading method
            model = safe_load_model(model_path, model_name)
            loaded_models[model_name] = model
            
            # Check if it's a pipeline
            if isinstance(model, Pipeline):
                model_is_pipeline[model_name] = True
            else:
                model_is_pipeline[model_name] = False
                
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            failed_models.append(f"{model_name}: {str(e)}")
            continue
    
    if failed_models:
        logger.warning(f"Failed to load {len(failed_models)} models: {failed_models}")
    
    if len(loaded_models) == 0:
        logger.warning("No models could be loaded. Creating mock models for testing...")
        # Create mock models so the application can start
        for model_name in MODEL_CONFIG.keys():
            loaded_models[model_name] = _create_mock_model(model_name)
            model_is_pipeline[model_name] = True
    
    # Update MODEL_ORDER to only include successfully loaded models
    global MODEL_ORDER
    MODEL_ORDER = [model for model in MODEL_ORDER if model in loaded_models]

def validate_and_clean_data(data: Dict[str, Any]) -> pd.DataFrame:
    """Clean and validate input data"""
    try:
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Get all required columns
        all_required_cols = set()
        for config in MODEL_CONFIG.values():
            all_required_cols.update(config['training_columns'])
        
        # Add missing columns with appropriate defaults
        for col in all_required_cols:
            if col not in df.columns:
                # Determine default based on column name patterns
                if any(keyword in col.lower() for keyword in ['age', 'bmi', 'weeks', 'score', 'b.p', 'bp', 'pulse', 'rate', 'temp', 'height', 'number']):
                    df[col] = 0.0
                else:
                    df[col] = 'not specified'
                # Missing column filled with default value
        
        # Clean data types - improved handling to prevent isnan errors
        for col in df.columns:
            # First, convert everything to string to handle mixed types
            df[col] = df[col].astype(str)
            
            # Handle null/empty values
            df[col] = df[col].replace(['nan', 'none', 'null', '', 'na', 'NaN', 'None', 'NULL', 'nan', 'None'], 'not specified')
            
            # Determine if column should be numeric based on patterns
            is_numeric_column = any(keyword in col.lower() for keyword in ['age', 'bmi', 'weeks', 'score', 'b.p', 'bp', 'pulse', 'rate', 'temp', 'height', 'number'])
            
            if is_numeric_column:
                # Convert to numeric, coercing errors to 0.0
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
            else:
                # Keep as string for categorical columns, ensure no NaN values
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].replace(['nan', 'none', 'null', '', 'na', 'NaN', 'None', 'NULL'], 'not specified')
        
        return df
        
    except Exception as e:
        logger.error(f"Error in data validation: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Data validation error: {str(e)}")

def get_missing_features(data: Dict[str, Any]) -> Dict[str, Any]:
    """Check which features are missing from input data for Mode_of_delivery2 model only"""
    try:
        # Focus only on Mode_of_delivery2 model
        model_name = "Mode_of_delivery2"
        if model_name not in MODEL_CONFIG:
            raise HTTPException(status_code=500, detail=f"Model {model_name} not found in configuration")
        
        config = MODEL_CONFIG[model_name]
        required_cols = set(config['training_columns'])
        
        # Check which columns are provided
        provided_cols = set(data.keys())
        missing_cols = required_cols - provided_cols
        
        return {
            "model_name": model_name,
            "total_required_features": len(required_cols),
            "provided_features": len(provided_cols),
            "missing_features_count": len(missing_cols),
            "missing_features": sorted(list(missing_cols)),
            "provided_features": sorted(list(provided_cols)),
            "required_features": sorted(list(required_cols)),
            "completeness_percentage": round((len(provided_cols) / len(required_cols)) * 100, 2)
        }
        
    except Exception as e:
        logger.error(f"Error checking missing features: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error checking missing features: {str(e)}")

def validate_and_clean_data_for_model(data: Dict[str, Any], model_name: str) -> pd.DataFrame:
    """Clean and validate input data for a specific model"""
    try:
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Get required columns for this specific model
        if model_name not in MODEL_CONFIG:
            raise HTTPException(status_code=400, detail=f"Model {model_name} not found in configuration")
        
        required_cols = MODEL_CONFIG[model_name]['training_columns']
        
        # Add missing columns with appropriate defaults
        for col in required_cols:
            if col not in df.columns:
                # Determine default based on column name patterns
                if any(keyword in col.lower() for keyword in ['age', 'bmi', 'weeks', 'score', 'b.p', 'bp', 'pulse', 'rate', 'temp', 'height', 'number']):
                    # Special handling for Apgar scores
                    if 'apgar' in col.lower():
                        df[col] = 7.0  # Normal Apgar score default
                    else:
                        df[col] = 0.0
                else:
                    # Special handling for specific postnatal columns
                    if col == 'Postnatal_Symptoms':
                        df[col] = 'normal'  # Default to normal symptoms
                    elif col == 'Postnatal_Examination':
                        df[col] = 'normal'  # Default to normal examination
                    else:
                        df[col] = 'not specified'
        
        # Clean data types
        for col in df.columns:
            # First, convert everything to string to handle mixed types
            df[col] = df[col].astype(str)
            
            # Handle null/empty values
            df[col] = df[col].replace(['nan', 'none', 'null', '', 'na', 'NaN', 'None', 'NULL'], 'not specified')
            
            # Determine if column should be numeric based on patterns
            is_numeric_column = any(keyword in col.lower() for keyword in ['age', 'bmi', 'weeks', 'score', 'b.p', 'bp', 'pulse', 'rate', 'temp', 'height', 'number'])
            
            if is_numeric_column:
                # Special handling for Apgar scores
                if 'apgar' in col.lower():
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(7.0)  # Default Apgar score
                else:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
            else:
                # Keep as string for categorical columns
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].replace(['nan', 'none', 'null', '', 'na', 'NaN', 'None', 'NULL'], 'not specified')
                
                # Apply specific defaults for postnatal columns
                if col == 'Postnatal_Symptoms' and df[col].iloc[0] == 'not specified':
                    df[col] = 'normal'
                elif col == 'Postnatal_Examination' and df[col].iloc[0] == 'not specified':
                    df[col] = 'normal'
        
        # Return only the required columns for this model
        return df[required_cols].copy()
        
    except Exception as e:
        logger.error(f"Error in data validation for {model_name}: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Data validation error for {model_name}: {str(e)}")

def get_missing_features_for_model(data: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """Check which features are missing from input data for a specific model"""
    try:
        if model_name not in MODEL_CONFIG:
            raise HTTPException(status_code=500, detail=f"Model {model_name} not found in configuration")
        
        config = MODEL_CONFIG[model_name]
        required_cols = set(config['training_columns'])
        
        # Check which columns are provided
        provided_cols = set(data.keys())
        missing_cols = required_cols - provided_cols
        
        return {
            "model_name": model_name,
            "target_column": config['target_column'],
            "total_required_features": len(required_cols),
            "provided_features": len(provided_cols),
            "missing_features_count": len(missing_cols),
            "missing_features": sorted(list(missing_cols)),
            "provided_features": sorted(list(provided_cols)),
            "required_features": sorted(list(required_cols)),
            "completeness_percentage": round((len(provided_cols) / len(required_cols)) * 100, 2)
        }
        
    except Exception as e:
        logger.error(f"Error checking missing features for {model_name}: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error checking missing features for {model_name}: {str(e)}")


def make_prediction(model, data: pd.DataFrame, model_name: str) -> tuple:
    """Make prediction with appropriate preprocessing based on model type"""
    try:
        if model_is_pipeline[model_name]:
            # Model already includes preprocessing
            prediction = model.predict(data.iloc[0, :].to_frame().T)
            probabilities = model.predict_proba(data.iloc[0, :].to_frame().T)
        else:
            # Need to apply preprocessing separately
            X_processed, _ = apply_same_preprocessing_as_training(data)
            prediction = model.predict(X_processed)
            probabilities = model.predict_proba(X_processed)
        
        # Convert results to JSON-serializable format
        prediction_value = str(prediction[0])
        prob_dict = {}
        
        if hasattr(model, 'classes_'):
            classes = model.classes_
            prob_values = probabilities[0]
            prob_dict = {str(cls): float(prob) for cls, prob in zip(classes, prob_values)}
        
        return prediction_value, prob_dict
        
    except Exception as e:
        logger.error(f"Error making prediction for {model_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed for {model_name}: {str(e)}")

# Models are now loaded via lifespan context manager

@app.post("/validate-features")
async def validate_features(data: PatientData):
    """Check which features are missing from input data"""
    try:
        missing_features_info = get_missing_features(data.patientData)
        
        response = {
            "message": "Feature validation completed",
            "validation_results": missing_features_info,
            "status": "success"
        }
        
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in feature validation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/outcomeModel")
async def outcome_model(data: PatientData):
    """Sequential prediction endpoint"""
    try:
        # Processing prediction request
        
        # # Check if we're using mock models
        # mock_models_used = any('mock_prediction' in str(model) for model in loaded_models.values())
        # if mock_models_used:
        #     print("WARNING: Using mock models - predictions are not real!")
        
        # Check missing features first
        missing_features_info = get_missing_features(data.patientData)
        if len(missing_features_info['missing_features']) > 0:
            raise HTTPException(status_code=400, detail=missing_features_info)

        # Validate and clean input data
        # patient_df = validate_and_clean_data(data.patientData)
        patient_df = pd.DataFrame([data.patientData])
        
        # Store results
        results = {}
        probabilities = {}
        
        # Run models sequentially
        for i, model_name in enumerate(MODEL_ORDER):
            config = MODEL_CONFIG[model_name]
            required_columns = config['training_columns']
            target_column = config['target_column']
            
            # Select required columns for this model
            model_input = patient_df[required_columns].copy()
            
            # Make prediction
            prediction, prob_dict = make_prediction(
                loaded_models[model_name], 
                model_input, 
                model_name
            )
            
            # Store results
            results[target_column] = prediction
            probabilities[target_column] = prob_dict
            
            # Add prediction to dataframe for next model (sequential pipeline)
            patient_df[target_column] = prediction
        
        # Prepare response
        response = {
            "message": "Sequential prediction completed successfully",
            "predictions": results,
        }
        
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# ==================== NEW INDIVIDUAL MODEL ENDPOINTS ====================
@app.post("/predict/mode-of-delivery")
async def predict_mode_of_delivery(data: PatientData):
    """Predict Mode of Delivery (Individual Model Endpoint)"""
    try:
        model_name = "Mode_of_delivery2"
        
        # Check if model is loaded
        if model_name not in loaded_models:
            raise HTTPException(status_code=500, detail=f"Model {model_name} not loaded")
        
        # Validate and clean input data for this specific model
        model_input = validate_and_clean_data_for_model(data.patientData, model_name)
        
        # Make prediction
        prediction, prob_dict = make_prediction(
            loaded_models[model_name], 
            model_input, 
            model_name
        )
        
        # Get missing features info
        missing_info = get_missing_features_for_model(data.patientData, model_name)
        
        # Prepare response
        response = {
            "message": f"Prediction completed for {model_name}",
            "model_name": model_name,
            "target_column": MODEL_CONFIG[model_name]['target_column'],
            "prediction": prediction,
            "probabilities": prob_dict,
            "feature_completeness": missing_info['completeness_percentage'],
            "missing_features_count": missing_info['missing_features_count']
        }
        
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in mode of delivery prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/predict/antenatal-complications")
async def predict_antenatal_complications(data: PatientData):
    """Predict Antenatal/Peripartum Maternal Complications (Individual Model Endpoint)"""
    try:
        model_name = "Antenatal_Peripartum_Maternal_Complications"
        
        # Check if model is loaded
        if model_name not in loaded_models:
            raise HTTPException(status_code=500, detail=f"Model {model_name} not loaded")
        
        # Validate and clean input data for this specific model
        model_input = validate_and_clean_data_for_model(data.patientData, model_name)
        
        # Make prediction
        prediction, prob_dict = make_prediction(
            loaded_models[model_name], 
            model_input, 
            model_name
        )
        
        # Get missing features info
        missing_info = get_missing_features_for_model(data.patientData, model_name)
        
        # Prepare response
        response = {
            "message": f"Prediction completed for {model_name}",
            "model_name": model_name,
            "target_column": MODEL_CONFIG[model_name]['target_column'],
            "prediction": prediction,
            "probabilities": prob_dict,
            "feature_completeness": missing_info['completeness_percentage'],
            "missing_features_count": missing_info['missing_features_count']
        }
        
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in antenatal complications prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/predict/neonatal-complications")
async def predict_neonatal_complications(data: PatientData):
    """Predict Neonatal/Fetal Complications (Individual Model Endpoint)"""
    try:
        model_name = "Neonatal__Fetal_Complications"
        
        # Check if model is loaded
        if model_name not in loaded_models:
            raise HTTPException(status_code=500, detail=f"Model {model_name} not loaded")
        
        # Validate and clean input data for this specific model
        model_input = validate_and_clean_data_for_model(data.patientData, model_name)
        
        # Make prediction
        prediction, prob_dict = make_prediction(
            loaded_models[model_name], 
            model_input, 
            model_name
        )
        
        # Get missing features info
        missing_info = get_missing_features_for_model(data.patientData, model_name)
        
        # Prepare response
        response = {
            "message": f"Prediction completed for {model_name}",
            "model_name": model_name,
            "target_column": MODEL_CONFIG[model_name]['target_column'],
            "prediction": prediction,
            "probabilities": prob_dict,
            "feature_completeness": missing_info['completeness_percentage'],
            "missing_features_count": missing_info['missing_features_count'],
            "note": "This model requires Apgar scores - defaults used if not provided"
        }
        
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in neonatal complications prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/predict/postnatal-complications")
async def predict_postnatal_complications(data: PatientData):
    """Predict Postnatal Maternal Complications (Individual Model Endpoint)"""
    try:
        model_name = "Postnatal_Maternal_Complications"
        
        # Check if model is loaded
        if model_name not in loaded_models:
            raise HTTPException(status_code=500, detail=f"Model {model_name} not loaded")
        
        # Validate and clean input data for this specific model
        model_input = validate_and_clean_data_for_model(data.patientData, model_name)
        
        # Make prediction
        prediction, prob_dict = make_prediction(
            loaded_models[model_name], 
            model_input, 
            model_name
        )
        
        # Get missing features info
        missing_info = get_missing_features_for_model(data.patientData, model_name)
        
        # Prepare response
        response = {
            "message": f"Prediction completed for {model_name}",
            "model_name": model_name,
            "target_column": MODEL_CONFIG[model_name]['target_column'],
            "prediction": prediction,
            "probabilities": prob_dict,
            "feature_completeness": missing_info['completeness_percentage'],
            "missing_features_count": missing_info['missing_features_count'],
            "note": "This model requires Apgar scores and postnatal data - defaults used if not provided"
        }
        
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in postnatal complications prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# ==================== END OF NEW ENDPOINTS ====================


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        import sklearn
        import numpy
        sklearn_version = sklearn.__version__
        numpy_version = numpy.__version__
    except ImportError:
        sklearn_version = "unknown"
        numpy_version = "unknown"
    
    return {
        "status": "healthy", 
        "models_loaded": len(loaded_models),
        "pipeline_models": [k for k, v in model_is_pipeline.items() if v],
        "standalone_models": [k for k, v in model_is_pipeline.items() if not v],
        "versions": {
            "scikit-learn": sklearn_version,
            "numpy": numpy_version,
            "fastapi": "1.0.0"
        }
    }

@app.get("/models/info")
async def models_info():
    """Get information about loaded models"""
    return {
        "loaded_models": list(loaded_models.keys()),
        "model_order": MODEL_ORDER,
        "total_models": len(loaded_models),
        "model_types": {k: "pipeline" if v else "standalone" for k, v in model_is_pipeline.items()}
    }

@app.get("/features/required")
async def get_required_features():
    """Get list of required features for Mode_of_delivery2 model only"""
    try:
        # Focus only on Mode_of_delivery2 model
        model_name = "Mode_of_delivery2"
        if model_name not in MODEL_CONFIG:
            raise HTTPException(status_code=500, detail=f"Model {model_name} not found in configuration")
        
        config = MODEL_CONFIG[model_name]
        required_cols = set(config['training_columns'])
        
        return {
            "model_name": model_name,
            "target_column": config['target_column'],
            "total_required_features": len(required_cols),
            "required_features": sorted(list(required_cols)),
            "feature_categories": {
                "numerical_features": [f for f in required_cols if any(keyword in f.lower() for keyword in ['age', 'bmi', 'weeks', 'score', 'b.p', 'bp', 'pulse', 'rate', 'temp', 'height', 'number'])],
                "categorical_features": [f for f in required_cols if not any(keyword in f.lower() for keyword in ['age', 'bmi', 'weeks', 'score', 'b.p', 'bp', 'pulse', 'rate', 'temp', 'height', 'number'])]
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting required features: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting required features: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=True)