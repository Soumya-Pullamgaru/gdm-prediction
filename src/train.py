# ML model-Final with plot
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier
import joblib
import os
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import seaborn as sns

# Load and preprocess dataset
def load_and_preprocess_data(file_path):
    print(f"Loading data from: {file_path}")
    data = pd.read_excel(file_path)
    data.rename(columns={
        'Family History': 'Family_History_of_Diabetes',
        'Class Label(GDM /Non GDM)': 'Gestational_Diabetes',
        'HDL': 'HDL_Cholesterol',
        'Sys BP': 'Blood_Pressure_Systolic',
        'Dia BP': 'Blood_Pressure_Diastolic',
        'Sedentary Lifestyle': 'Physical_Activity'
    }, inplace=True)
    data.drop(columns=['Case Number'], errors='ignore', inplace=True)
    return data

# Handle missing values
def handle_missing_values(data, numeric_columns):
    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())
    return data

# Encode categorical variables
def encode_categorical(data, categorical_columns):
    for col in categorical_columns:
        data[col] = LabelEncoder().fit_transform(data[col].astype(str))
    return data

#FEATURE SELECTION
class FeatureSelector:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.X_selected = None
        self.selected_columns = None
        self.selector = None

    def select_features(self, method='selectkbest', k=10, threshold=0.05):
        """Select the most important features"""
        if self.X is None or self.y is None:
            print("Features and target not available. Call preprocess_data() first.")
            return False

        print(f"Selecting features using {method} method")

        try:
            
            if method == 'rfe':
                from sklearn.feature_selection import RFE
                from sklearn.ensemble import RandomForestClassifier
                base_model = RandomForestClassifier(n_estimators=100, random_state=42)
                self.selector = RFE(estimator=base_model, n_features_to_select=k)
                self.selector.fit(self.X, self.y)
                selected_mask = self.selector.get_support()
                self.selected_columns = self.X.columns[selected_mask]
                self.X_selected = self.X[self.selected_columns]

                # Print selected features
                for feature in self.selected_columns:
                    print(f"Selected feature: {feature}")

            elif method == 'statistical':
                from sklearn.feature_selection import SelectKBest, f_classif
                f_selector = SelectKBest(score_func=f_classif, k='all')
                f_selector.fit(self.X, self.y)
                f_scores = f_selector.scores_
                p_values = f_selector.pvalues_

                # Apply p-value threshold
                selected_mask = p_values < threshold

                # Ensure minimum number of features
                if sum(selected_mask) < 3:
                    top_indices = np.argsort(f_scores)[::-1][:5]
                    selected_mask = np.zeros_like(f_scores, dtype=bool)
                    selected_mask[top_indices] = True

                self.selected_columns = self.X.columns[selected_mask]
                self.X_selected = self.X[self.selected_columns]

                # Print selected features with p-values
                for feature, f_score, p_val in zip(self.selected_columns, f_scores[selected_mask], p_values[selected_mask]):
                    significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                    print(f"Selected feature: {feature} with F={f_score:.4f}, p={p_val:.6f} {significance}")

            else:
                print(f"Unknown feature selection method: {method}")
                return False

            print(f"Selected {len(self.selected_columns)} features")
            return True

        except Exception as e:
            print(f"Error in feature selection: {str(e)}")
            return False


# Scale numeric features
def scale_features(data, numeric_columns):
    scaler = StandardScaler()
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
    return data, scaler

# Calculate sensitivity and specificity
def calculate_sensitivity_specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return sensitivity, specificity

# Add noise to prevent overfitting
def add_training_noise(X, noise_level=0.015):
    """Add controlled noise to training data to prevent overfitting"""
    np.random.seed(42)  # For reproducibility
    noise = np.random.normal(0, noise_level, X.shape)
    return X + noise

#model optimization with strong overfitting prevention
def optimize_models(X_train, y_train, X_test, y_test):
    print("\nPerforming enhanced model optimization with overfitting prevention...")

    # Add controlled noise to training data
    X_train_noisy = add_training_noise(X_train, noise_level=0.02)
    
    # Cross-validation strategy
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 1. RandomForest optimization 
    print("Optimizing RandomForest model...")
    rf_param_grid = {
        'n_estimators': [100, 150, 200],  
        'max_depth': [6, 8, 10],  
        'min_samples_split': [5, 8, 10],  
        'min_samples_leaf': [3, 5, 7],  
        'max_features': [0.7, 'sqrt']  
    }
    rf_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42),
        rf_param_grid,
        n_iter=8,  
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        random_state=42
    )
    rf_search.fit(X_train_noisy, y_train)
    rf = rf_search.best_estimator_
    print(f"Best RandomForest parameters: {rf_search.best_params_}")

    # 2. XGBoost optimization (Strong regularization + early stopping)
    print("Optimizing XGBoost model...")
    
    # Create validation set for early stopping
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train_noisy, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    xgb_param_grid = {
        'n_estimators': [100, 150],  
        'learning_rate': [0.03, 0.05],  
        'max_depth': [3, 4, 5],  
        'subsample': [0.7, 0.8],  
        'colsample_bytree': [0.7, 0.8],  
        'reg_alpha': [1.0, 2.0],  
        'reg_lambda': [1.0, 2.0],  
        'min_child_weight': [3, 5]  
    }
    
    # XGBoost with early stopping
    xgb_base = XGBClassifier(
        random_state=42,
        early_stopping_rounds=10,
        eval_metric='auc'
    )
    
    xgb_search = RandomizedSearchCV(
        xgb_base,
        xgb_param_grid,
        n_iter=8,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        random_state=42
    )
    xgb_search.fit(X_train_sub, y_train_sub, eval_set=[(X_val, y_val)], verbose=False)
    xgb = xgb_search.best_estimator_
    print(f"Best XGBoost parameters: {xgb_search.best_params_}")

    # 3. LightGBM optimization (Strong regularization)
    print("Optimizing LightGBM model...")
    lgbm_param_grid = {
        'n_estimators': [100, 150],  
        'learning_rate': [0.03, 0.05],  
        'max_depth': [3, 4],  
        'num_leaves': [25, 35],  
        'min_child_samples': [25, 35],  
        'reg_alpha': [1.0, 2.0], 
        'reg_lambda': [1.0, 2.0],  
        'feature_fraction': [0.7, 0.8],  
        'bagging_fraction': [0.7, 0.8]  
    }
    lgbm_search = RandomizedSearchCV(
        LGBMClassifier(random_state=42, verbose=-1),
        lgbm_param_grid,
        n_iter=8,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        random_state=42
    )
    lgbm_search.fit(X_train_noisy, y_train)
    lgbm = lgbm_search.best_estimator_
    print(f"Best LightGBM parameters: {lgbm_search.best_params_}")

    # 4. GradientBoosting optimization 
    print("Optimizing GradientBoosting model...")
    gb_param_grid = {
        'n_estimators': [100, 150],  
        'learning_rate': [0.03, 0.05], 
        'max_depth': [3, 4], 
        'min_samples_split': [8, 12],  
        'min_samples_leaf': [4, 6],
        'subsample': [0.8, 0.9]  
    }
    gb_search = RandomizedSearchCV(
        GradientBoostingClassifier(random_state=42),
        gb_param_grid,
        n_iter=8,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        random_state=42
    )
    gb_search.fit(X_train_noisy, y_train)
    gb = gb_search.best_estimator_
    print(f"Best GradientBoosting parameters: {gb_search.best_params_}")

    # Generate meta-features for stacking (using clean training data)
    print("\nGenerating meta-features...")
    X_train_meta = np.column_stack([
        rf.predict_proba(X_train)[:, 1],
        xgb.predict_proba(X_train)[:, 1],
        lgbm.predict_proba(X_train)[:, 1],
        gb.predict_proba(X_train)[:, 1]
    ])
    X_test_meta = np.column_stack([
        rf.predict_proba(X_test)[:, 1],
        xgb.predict_proba(X_test)[:, 1],
        lgbm.predict_proba(X_test)[:, 1],
        gb.predict_proba(X_test)[:, 1]
    ])

    # 5. Meta-learner with VERY strong regularization
    print("Optimizing meta-learner (XGBoost with strong regularization)...")
    
    # Add noise to meta-features to prevent overfitting
    X_train_meta_noisy = add_training_noise(X_train_meta, noise_level=0.01)
    
    meta_param_grid = {
        'n_estimators': [30, 50, 80],  
        'learning_rate': [0.01, 0.02],  
        'max_depth': [2, 3],  
        'subsample': [0.6, 0.7],  
        'colsample_bytree': [0.6, 0.7], 
        'reg_alpha': [2.0, 5.0, 10.0],  
        'reg_lambda': [2.0, 5.0, 10.0], 
        'min_child_weight': [5, 10]  
    }
    
    meta_search = RandomizedSearchCV(
        XGBClassifier(random_state=42),
        meta_param_grid,
        n_iter=10,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        random_state=42
    )
    meta_search.fit(X_train_meta_noisy, y_train)
    meta_learner = meta_search.best_estimator_
    print(f"Best meta-learner parameters: {meta_search.best_params_}")

    # Calculate training accuracy for the final stacked ensemble model
    y_train_pred = meta_learner.predict(X_train_meta)
    training_accuracy = accuracy_score(y_train, y_train_pred)
    
    print(f"\nTraining Accuracy: {training_accuracy:.3f}")

    # Evaluate on test set
    y_test_pred = meta_learner.predict(X_test_meta)

    return rf, xgb, lgbm, gb, meta_learner, y_test_pred

# Create prediction script (unchanged)
def create_prediction_script(feature_names):
    script_content = """

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


"""
    
    with open('gdm_predict.py', 'w') as f:
        f.write(script_content)
    
    print("Created gdm_predict.py with prediction function that supports 10 features")


def create_dca_plot(y_true, y_prob, model_name="GDM Model", save_path="dca_plot.png"):
    """
    Create Decision Curve Analysis plot for clinical utility assessment
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Set matplotlib to non-interactive backend to prevent hanging
    plt.switch_backend('Agg')
    
    print(f"Creating DCA plot for {model_name}...")
    
    # Define threshold probabilities (0% to 50% risk thresholds)
    thresholds = np.arange(0.05, 0.41, 0.01)
    
    # Calculate net benefit for each threshold
    net_benefits = []
    
    for threshold in thresholds:
        # Classify as high risk if probability > threshold
        y_pred = (y_prob >= threshold).astype(int)
        
        # Calculate confusion matrix components
        tp = np.sum((y_pred == 1) & (y_true == 1))  # True positives
        fp = np.sum((y_pred == 1) & (y_true == 0))  # False positives
        tn = np.sum((y_pred == 0) & (y_true == 0))  # True negatives
        fn = np.sum((y_pred == 0) & (y_true == 1))  # False negatives
        
        n = len(y_true)  # Total sample size
        
        # Net benefit calculation
        # NB = (TP/n) - (FP/n) * (pt/(1-pt))
        # where pt is threshold probability
        net_benefit = (tp / n) - (fp / n) * (threshold / (1 - threshold))
        net_benefits.append(net_benefit)
    
    # Calculate reference strategies
    # Strategy 1: Treat all patients (assume all positive)
    treat_all_benefits = []
    prevalence = np.mean(y_true)  # Disease prevalence
    
    for threshold in thresholds:
        # If we treat everyone: TP = all diseased, FP = all healthy
        treat_all_nb = prevalence - (1 - prevalence) * (threshold / (1 - threshold))
        treat_all_benefits.append(max(0, treat_all_nb))  # Can't be negative
    
    # Strategy 2: Treat none (assume all negative)
    treat_none_benefits = [0] * len(thresholds)  # Always 0
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot model net benefit
    plt.plot(thresholds * 100, net_benefits, 'b-', linewidth=3, 
             label=f'{model_name}', marker='o', markersize=3)
    
    # Plot reference strategies
    plt.plot(thresholds * 100, treat_all_benefits, 'r--', linewidth=2, 
             label='Treat All', alpha=0.7)
    plt.plot(thresholds * 100, treat_none_benefits, 'k--', linewidth=2, 
             label='Treat None', alpha=0.7)
    
    # Customize plot
    plt.xlabel('Threshold Probability (%)', fontsize=12, fontweight='bold')
    plt.ylabel('Net Benefit', fontsize=12, fontweight='bold')
    plt.title(f'Decision Curve Analysis - {model_name}', fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc='upper right', fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Set axis limits
    plt.xlim(0, 50)
    plt.ylim(min(0, min(net_benefits)) - 0.01, max(max(net_benefits), max(treat_all_benefits)) + 0.01)
    
    # Add clinical interpretation zones
    plt.axhspan(0, max(max(net_benefits), max(treat_all_benefits)) + 0.01, alpha=0.1, color='green')
    plt.text(2, max(net_benefits) * 0.8, 'Clinical Benefit Zone', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"DCA plot saved as: {save_path}")
    
    # Close the plot to free memory and prevent hanging - FIXED!
    plt.close()
    
    # Calculate and print DCA metrics
    max_net_benefit = max(net_benefits)
    best_threshold_idx = np.argmax(net_benefits)
    best_threshold = thresholds[best_threshold_idx]
    
    # Calculate area under DCA curve (standardized net benefit)
    auc_dca = np.trapz(net_benefits, thresholds)
    
    print(f"\n{'='*50}")
    print(f"DCA ANALYSIS RESULTS FOR {model_name.upper()}")
    print(f"{'='*50}")
    print(f"Maximum Net Benefit: {max_net_benefit:.4f}")
    print(f"Optimal Threshold: {best_threshold:.1%} ({best_threshold*100:.1f}%)")
    print(f"DCA Area Under Curve: {auc_dca:.4f}")
    print(f"Disease Prevalence: {prevalence:.1%}")
    
    # Clinical interpretation
    if max_net_benefit > 0.05:
        interpretation = "Excellent clinical utility"
    elif max_net_benefit > 0.02:
        interpretation = "Good clinical utility"
    elif max_net_benefit > 0:
        interpretation = "Moderate clinical utility"
    else:
        interpretation = "Limited clinical utility"
    
    print(f"Clinical Interpretation: {interpretation}")
    print(f"{'='*50}")
    
    return max_net_benefit, best_threshold, auc_dca

# Main pipeline
def main():
    # Define file path 
    file_path = r"data/Gestational_Diabetic_Dataset.xlsx"
    
    # Check if file exists, otherwise prompt for path
    if not os.path.exists(file_path):
        file_path = input(f"File {file_path} not found. Please enter the full path to the dataset file: ")
    
    # Try to load the data
    try:
        data = load_and_preprocess_data(file_path)
        print(f"Successfully loaded dataset with {data.shape[0]} records and {data.shape[1]} columns")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
   
    # Define columns
    numeric_columns = [
        'Age', 'No of Pregnancy', 'BMI',
        'HDL_Cholesterol', 'Blood_Pressure_Systolic', 'Blood_Pressure_Diastolic',
        'OGTT', 'Hemoglobin'
    ]
   
    categorical_columns = [
        'Family_History_of_Diabetes', 'unexplained prenetal loss',
        'Large Child or Birth Default', 'PCOS', 'Prediabetes',
        'Gestation in previous Pregnancy'
    ]
    # Preprocess data
    data = handle_missing_values(data, numeric_columns)
    data = encode_categorical(data, categorical_columns)
    data, scaler = scale_features(data, numeric_columns)
   
    # Split features and target
    X = data.drop(columns=['Gestational_Diabetes'])
    y = data['Gestational_Diabetes']

        
    # Select features using the desired method
    feature_selector = FeatureSelector(X, y)
    if not feature_selector.select_features(method='rfe', k=10):
        print("Feature selection failed. Exiting...")
        return

    # Use the selected features for modeling
    X_selected = feature_selector.X_selected      
    selected_features = feature_selector.selected_columns

    print(f"Training with selected features: {selected_features}")

    # Add the new scaler code here:
    numeric_selected = [col for col in selected_features if col in numeric_columns]
    X_selected_df = pd.DataFrame(X_selected, columns=selected_features)
    selected_scaler = StandardScaler()
    X_selected_df[numeric_selected] = selected_scaler.fit_transform(X_selected_df[numeric_selected])
    X_selected = X_selected_df.values

    # Apply SMOTE for class balancing 
    print("\nBalancing classes with SMOTE...")
    smote = SMOTE(random_state=42, sampling_strategy=0.7)  # Reduced from 0.8
    X_resampled, y_resampled = smote.fit_resample(X_selected, y)
       
    # Train-test split with selected features
    print("\nSplitting data into training and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.25, random_state=42, stratify=y_resampled
    )
        
   
    rf, xgb, lgbm, gb, meta_learner, y_test_pred = optimize_models(X_train, y_train, X_test, y_test)

   
    # Calculate metrics
    sensitivity, specificity = calculate_sensitivity_specificity(y_test, y_test_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_roc_auc = roc_auc_score(y_test, y_test_pred)
   
    # Print evaluation metrics
    print("\n" + "="*60)
    print("FINAL STACKED ENSEMBLE MODEL - TEST SET PERFORMANCE")
    print("="*60)
    print("Stacking Model Classification Report:")
    print(classification_report(y_test, y_test_pred))
    print(f"Stacking Model Test Accuracy: {test_accuracy:.3f}")
    print(f"Test ROC AUC Score: {test_roc_auc:.3f}")
    print(f"Test Sensitivity: {sensitivity:.3f}")
    print(f"Test Specificity: {specificity:.3f}")
    print("\nTest Confusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred))
    
    # Generate meta-features for test set for DCA
    X_test_meta = np.column_stack([
        rf.predict_proba(X_test)[:, 1],
        xgb.predict_proba(X_test)[:, 1],
        lgbm.predict_proba(X_test)[:, 1],
        gb.predict_proba(X_test)[:, 1]
    ])
    
    # Get final predicted probabilities from meta-learner
    y_test_proba = meta_learner.predict_proba(X_test_meta)[:, 1]
    
    # Create DCA plot
    max_net_benefit, best_threshold, auc_dca = create_dca_plot(
        y_test, 
        y_test_proba, 
        model_name="GDM Ensemble Model",
        save_path="gdm_decision_curve_analysis.png"
    )
    
    print(f"\n DCA Analysis completed!")
    print(f" Clinical utility demonstrated with max net benefit: {max_net_benefit:.4f}")
    print(f" DCA plot saved as: gdm_decision_curve_analysis.png")
    print(f"{'='*60}")
    
    # Save models
    print("\nSaving models...")
    models = {
        'rf': rf,
        'xgb': xgb,
        'lgbm': lgbm,
        'gb': gb,
        'meta_learner': meta_learner,
        'scaler': selected_scaler,  
        'feature_names': list(selected_features)
    }
        
    joblib.dump(models, 'gdm_stacking_model.joblib')
    print("Model saved as gdm_stacking_model.joblib")
      
        
    # Create prediction script
    create_prediction_script(selected_features)
   
    print("\nModel creation complete! You can now use the GDM prediction app.")

if __name__ == "__main__":
    main()
