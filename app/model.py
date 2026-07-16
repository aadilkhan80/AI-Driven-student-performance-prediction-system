import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, classification_report
import joblib

class StudentPerformanceModel:
    def __init__(self, model_dir=None):
        if model_dir is None:
            # Save in the app directory under a "models" subfolder
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.model_dir = os.path.join(current_dir, 'models')
        else:
            self.model_dir = model_dir
            
        self.gpa_model_path = os.path.join(self.model_dir, 'gpa_model.joblib')
        self.risk_model_path = os.path.join(self.model_dir, 'risk_model.joblib')
        self.stats_path = os.path.join(self.model_dir, 'model_stats.json')
        
        self.gpa_pipeline = None
        self.risk_pipeline = None
        self.metrics = {}
        
        # Define features
        self.num_features = ['study_hours', 'attendance', 'sleep_hours', 'previous_grade', 
                             'tutoring', 'extracurriculars', 'internet_access']
        self.cat_features = ['parental_support', 'stress_level']
        
    def _build_preprocessor(self):
        # We specify categories to ensure consistent one-hot columns
        categories = [
            ['Low', 'Medium', 'High'],  # parental_support
            ['Low', 'Medium', 'High']   # stress_level
        ]
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.num_features),
                ('cat', OneHotEncoder(categories=categories, sparse_output=False), self.cat_features)
            ]
        )
        return preprocessor
        
    def train(self, data_df):
        """
        Trains both GPA regressor and Risk classifier, evaluations them,
        saves the pipeline models and metric statistics.
        """
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Separate features and targets
        X = data_df[self.num_features + self.cat_features]
        y_gpa = data_df['gpa']
        y_risk = data_df['risk_level']
        
        # 1. Train-test splits
        X_train, X_test, y_train_gpa, y_test_gpa = train_test_split(X, y_gpa, test_size=0.2, random_state=42)
        _, _, y_train_risk, y_test_risk = train_test_split(X, y_risk, test_size=0.2, random_state=42)
        
        # 2. GPA Regressor Pipeline
        gpa_preprocessor = self._build_preprocessor()
        self.gpa_pipeline = Pipeline([
            ('preprocessor', gpa_preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10))
        ])
        
        self.gpa_pipeline.fit(X_train, y_train_gpa)
        y_pred_gpa = self.gpa_pipeline.predict(X_test)
        
        # Calculate GPA regression metrics
        r2 = r2_score(y_test_gpa, y_pred_gpa)
        mse = mean_squared_error(y_test_gpa, y_pred_gpa)
        mae = np.mean(np.abs(y_test_gpa - y_pred_gpa))
        
        # 3. Risk Classifier Pipeline
        risk_preprocessor = self._build_preprocessor()
        self.risk_pipeline = Pipeline([
            ('preprocessor', risk_preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10))
        ])
        
        self.risk_pipeline.fit(X_train, y_train_risk)
        y_pred_risk = self.risk_pipeline.predict(X_test)
        
        # Calculate Risk classification metrics
        acc = accuracy_score(y_test_risk, y_pred_risk)
        
        # Save model state files
        joblib.dump(self.gpa_pipeline, self.gpa_model_path)
        joblib.dump(self.risk_pipeline, self.risk_model_path)
        
        # 4. Extract Feature Importances
        # Get feature names after preprocessing
        # Number features retain their name. For categoricals, we build them manually.
        encoded_cat_names = []
        encoder_categories = [['Low', 'Medium', 'High'], ['Low', 'Medium', 'High']]
        for i, feature in enumerate(self.cat_features):
            for val in encoder_categories[i]:
                encoded_cat_names.append(f"{feature}_{val}")
                
        all_feature_names = self.num_features + encoded_cat_names
        
        # Regressor importances
        gpa_importances = self.gpa_pipeline.named_steps['regressor'].feature_importances_
        feature_importance_gpa = {name: float(importance) for name, importance in zip(all_feature_names, gpa_importances)}
        
        # Classifier importances
        risk_importances = self.risk_pipeline.named_steps['classifier'].feature_importances_
        feature_importance_risk = {name: float(importance) for name, importance in zip(all_feature_names, risk_importances)}
        
        # Combine statistics
        self.metrics = {
            'gpa_metrics': {
                'r2_score': float(r2),
                'mean_squared_error': float(mse),
                'mean_absolute_error': float(mae)
            },
            'risk_metrics': {
                'accuracy': float(acc)
            },
            'feature_importances': {
                'gpa': feature_importance_gpa,
                'risk': feature_importance_risk
            },
            'dataset_size': len(data_df)
        }
        
        with open(self.stats_path, 'w') as f:
            json.dump(self.metrics, f, indent=4)
            
        print(f"Models trained and saved in: {self.model_dir}")
        return self.metrics
        
    def load(self):
        """
        Loads models and stats if they exist. Returns True if successful, False otherwise.
        """
        if (os.path.exists(self.gpa_model_path) and 
            os.path.exists(self.risk_model_path) and 
            os.path.exists(self.stats_path)):
            try:
                self.gpa_pipeline = joblib.load(self.gpa_model_path)
                self.risk_pipeline = joblib.load(self.risk_model_path)
                with open(self.stats_path, 'r') as f:
                    self.metrics = json.load(f)
                return True
            except Exception as e:
                print(f"Error loading models: {e}")
                return False
        return False
        
    def predict(self, student_data):
        """
        Runs GPA and Risk predictions for a single input dictionary
        student_data keys matching: study_hours, attendance, sleep_hours, previous_grade,
                                   tutoring, extracurriculars, internet_access, parental_support, stress_level
        """
        if self.gpa_pipeline is None or self.risk_pipeline is None:
            success = self.load()
            if not success:
                raise ValueError("Models are not trained or loaded. Train the models first.")
                
        # Format single input into a dataframe
        df = pd.DataFrame([student_data])
        
        # Predict
        pred_gpa = self.gpa_pipeline.predict(df)[0]
        pred_risk = self.risk_pipeline.predict(df)[0]
        
        return {
            'predicted_gpa': round(float(pred_gpa), 2),
            'predicted_risk': str(pred_risk)
        }
        
    def get_stats(self):
        if not self.metrics:
            self.load()
        return self.metrics
