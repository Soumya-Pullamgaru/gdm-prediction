# gdm_predict.py
import numpy as np
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load trained model and preprocessing objects
models = joblib.load('gdm_stacking_model.joblib')
scaler = models['scaler']
feature_names = models['feature_names']
print(f"Model loaded successfully. Expected feature names: {feature_names}")

# Define categorical features used in the model
categorical_features = ['Large Child or Birth Default', 'PCOS', 'Prediabetes', 'Gestation in previous Pregnancy']

# Function to preprocess user inputs before prediction
def preprocess_features(features):
    try:
        print(f"Preprocessing features: {features}")

        # Convert user input into dictionary
        feature_dict = {name: value for name, value in zip(feature_names, features)}

        # Convert categorical "Yes/No" values to 1/0
        for cat_feature in categorical_features:
            if cat_feature in feature_dict:
                feature_dict[cat_feature] = 1 if str(feature_dict[cat_feature]).strip().lower() == "yes" else 0

        # Create DataFrame to match expected feature format
        features_df = pd.DataFrame([{name: feature_dict.get(name, 0) for name in feature_names}])

        # Scale numeric features using the trained scaler
        numeric_columns = [col for col in feature_names if col not in categorical_features]
        features_df[numeric_columns] = scaler.transform(features_df[numeric_columns])

        return features_df.values, features_df

    except Exception as e:
        print(f"Error in feature preprocessing: {e}")
        return np.array(features).reshape(1, -1)

# Prediction function
 
def predict(features):
    
    print(f"Received {len(features)} features for prediction")
    print(f"Features received: {features}")
    
    try:
        # Preprocess user input
        features_array, features_df = preprocess_features(features)

        # Ensure required model components are available
        required_components = ['rf', 'xgb', 'lgbm', 'gb', 'meta_learner']
        missing_components = [comp for comp in required_components if comp not in models]

        if missing_components:
            raise ValueError(f"Missing model components: {missing_components}. Retrain the model.")

        # Load trained models
        rf = models['rf']
        xgb = models['xgb']
        lgbm = models['lgbm']
        gb = models['gb']
        meta_learner = models['meta_learner']

        # Get predictions from base models
        rf_pred = rf.predict_proba(features_array)[:, 1]
        xgb_pred = xgb.predict_proba(features_array)[:, 1]
        lgbm_pred = lgbm.predict_proba(features_df)[:, 1]  # LGBM requires DataFrame
        gb_pred = gb.predict_proba(features_array)[:, 1]

        # Stack predictions for meta-learner
        meta_features = np.column_stack([rf_pred, xgb_pred, lgbm_pred, gb_pred])

        # Compute meta_learner probability - THIS IS THE KEY LINE
        meta_proba = meta_learner.predict_proba(meta_features)[:, 1]

        threshold = 0.5
        
        # Apply threshold to get final prediction
        final_pred = (meta_proba > threshold).astype(int)

        print(f"Meta-learner probability: {meta_proba[0]:.4f}")
        print(f"Adaptive threshold: {threshold:.4f}")
        print(f"Final prediction: {final_pred[0]}")
        
        # Return both the binary prediction and the meta-learner probability
        return final_pred[0], float(meta_proba[0])

    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return 0, 0.0  # Default safe prediction if an error occurs


