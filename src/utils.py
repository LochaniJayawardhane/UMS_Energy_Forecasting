import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
import xgboost as xgb
import optuna
import os
import time
from typing import Optional

# Import task_tracker for progress updates
try:
    # When imported as a module
    from src.task_system import task_tracker, TaskState
except ModuleNotFoundError:
    # When run directly from src/ directory
    from task_system import task_tracker, TaskState

def optuna_progress_callback(study, trial, task_id: Optional[str] = None):
    """
    Callback function for Optuna that reports progress to the task_tracker.
    
    Args:
        study: Optuna study object
        trial: Current trial
        task_id: Task ID for reporting progress
    """
    if task_id is None:
        # Try to get task_id from environment in case it's available
        import dramatiq.middleware
        try:
            current_message = dramatiq.middleware.CurrentMessage.get_current_message()
            if current_message:
                task_id = current_message.message_id
        except:
            # No task context available
            return
    
    if task_id:
        # Calculate progress percentage - each trial is worth 1%
        n_trials = len(study.trials)
        total_trials = 50  # Hardcoded value matching the optimize call
        progress_percent = min(int((n_trials / total_trials) * 100), 99)
        
        # Add hyperparameter tuning progress to task tracker
        task_tracker.update_progress(
            task_id,
            TaskState.TRAINING_MODEL,
            50 + int(progress_percent * 0.35),  # Map 0-100% of Optuna to 50-85% overall progress
            f"Training model: Hyperparameter optimization in progress ({n_trials}/{total_trials} trials completed)",
            {
                "current_trial": n_trials,
                "total_trials": total_trials,
                "best_value": float(study.best_value) if hasattr(study, 'best_trial') and study.best_trial else None,
                "optimization_stage": "hyperparameter_tuning"
            }
        )

def create_features(df: pd.DataFrame, meter_id: str, meter_type: str = None) -> pd.DataFrame:
    """Create features for forecasting models."""
    df = df.copy()
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df['hour'] = df['DateTime'].dt.hour
    df['day'] = df['DateTime'].dt.day
    df['month'] = df['DateTime'].dt.month
    df['day_of_week'] = df['DateTime'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Initialize rolling temperature feature
    df['temp_rolling_mean_24h'] = df['Temperature'].rolling(window=24, min_periods=1).mean()
    
    # Handle consumption-based features
    has_consumption_data = ('Consumption' in df.columns and 
                           len(df['Consumption'].dropna()) > 0)
    
    if has_consumption_data:
        # Training scenario: Calculate actual lag and rolling features
        df['consumption_lag_24'] = df['Consumption'].shift(24)
        df['consumption_rolling_mean_24h'] = df['Consumption'].rolling(window=24, min_periods=1).mean()
        
        # Fill missing values with mean for training
        for col in ['consumption_lag_24', 'temp_rolling_mean_24h', 'consumption_rolling_mean_24h']:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mean())
    else:
        if meter_type == 'electricity':
            default_consumption = 104.87  # Electricity average from training data
        elif meter_type == 'water':
            default_consumption = 0.01    # Water average from training data
        else:
            # Fallback if meter_type not specified (backwards compatibility)
            default_consumption = 50.0
        
        df['consumption_lag_24'] = default_consumption
        df['consumption_rolling_mean_24h'] = default_consumption
        df['temp_rolling_mean_24h'] = df['temp_rolling_mean_24h'].fillna(df['temp_rolling_mean_24h'].mean())
    
   
    feature_columns = [
        'Temperature', 'hour', 'day', 'month', 'day_of_week', 'is_weekend',
        'consumption_lag_24', 'temp_rolling_mean_24h', 'consumption_rolling_mean_24h'
    ]
    return df[feature_columns]

def train_electricity_model(df, task_id: Optional[str] = None):
    """
    Complete training pipeline for electricity models, including:
    - Data preprocessing
    - Feature engineering
    - Hyperparameter optimization
    - Model training with early stopping
    """
    df = df.copy()
    # Preprocessing
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df = df.sort_values('DateTime')
    
    # Set DateTime as index for time-based operations
    df = df.set_index('DateTime')
    
    # Handle missing or zero temperature values
    df['Temperature'] = df['Temperature'].replace(0, np.nan)
    df['Temperature'] = df['Temperature'].interpolate(method='time')
    
    # Reset index to keep DateTime as a column for feature engineering
    df = df.reset_index()
    
    # Feature engineering
    features = create_features(df, meter_id="electricity", meter_type="electricity")
    
    # Scale numerical features with RobustScaler (specific to electricity)
    scaler = RobustScaler()
    numerical_cols = ['temp_rolling_mean_24h', 'consumption_lag_24', 'consumption_rolling_mean_24h']
    for col in numerical_cols:
        if col in features.columns:
            features[[col]] = scaler.fit_transform(features[[col]])
    
    X = features
    y = df['Consumption']
    
    # Split data for hyperparameter optimization
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Update task status if task_id is provided
    if task_id:
        task_tracker.update_progress(
            task_id, 
            TaskState.TRAINING_MODEL, 
            50, 
            "Training model: Starting hyperparameter optimization",
            {"stage": "hyperparameter_optimization"}
        )
    
    # Hyperparameter optimization with Optuna
    def objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
            'random_state': 42
        }
        model = xgb.XGBRegressor(**param)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error')
        return -cv_scores.mean()
    
    study = optuna.create_study(direction='minimize')
    
    # Set up the progress callback for Optuna
    callbacks = []
    if task_id:
        callbacks.append(lambda study, trial: optuna_progress_callback(study, trial, task_id))
    
    study.optimize(objective, n_trials=50, show_progress_bar=True, callbacks=callbacks)
    best_params = study.best_params
    
    # Update task status for final training stage
    if task_id:
        task_tracker.update_progress(
            task_id, 
            TaskState.TRAINING_MODEL, 
            85, 
            "Training model: Final training with best parameters",
            {
                "stage": "final_training", 
                "best_params": best_params
            }
        )
    
    # Train final model with early stopping
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    final_model = xgb.XGBRegressor(
        **best_params,
        early_stopping_rounds=50
    )
    
    eval_set = [(X_train_final, y_train_final), (X_val, y_val)]
    final_model.fit(
        X_train_final,
        y_train_final,
        eval_set=eval_set,
        verbose=False
    )
    
    return final_model

def train_water_model(df, task_id: Optional[str] = None):
    """
    Complete training pipeline for water models, including:
    - Data preprocessing
    - Feature engineering
    - Hyperparameter optimization
    - Model training with early stopping
    """
    df = df.copy()
    # Preprocessing
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df = df.sort_values('DateTime')
    
    # Set DateTime as index for time-based operations
    df = df.set_index('DateTime')
    
    # Handle missing or zero temperature values
    df['Temperature'] = df['Temperature'].replace(0, np.nan)
    df['Temperature'] = df['Temperature'].interpolate(method='time')
    
    # Reset index to keep DateTime as a column for feature engineering
    df = df.reset_index()
    
    # Feature engineering
    features = create_features(df, meter_id="water", meter_type="water")
    
    # Scale numerical features with StandardScaler (specific to water)
    scaler = StandardScaler()
    numerical_cols = ['temp_rolling_mean_24h', 'consumption_lag_24', 'consumption_rolling_mean_24h']
    for col in numerical_cols:
        if col in features.columns:
            features[[col]] = scaler.fit_transform(features[[col]])
    
    X = features
    y = df['Consumption']
    
    # Split data for hyperparameter optimization
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Update task status if task_id is provided
    if task_id:
        task_tracker.update_progress(
            task_id, 
            TaskState.TRAINING_MODEL, 
            50, 
            "Training model: Starting hyperparameter optimization",
            {"stage": "hyperparameter_optimization"}
        )
    
    # Hyperparameter optimization with Optuna
    def objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
            'random_state': 42
        }
        model = xgb.XGBRegressor(**param)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error')
        return -cv_scores.mean()
    
    study = optuna.create_study(direction='minimize')
    
    # Set up the progress callback for Optuna
    callbacks = []
    if task_id:
        callbacks.append(lambda study, trial: optuna_progress_callback(study, trial, task_id))
    
    study.optimize(objective, n_trials=50, show_progress_bar=True, callbacks=callbacks)
    best_params = study.best_params
    
    # Update task status for final training stage
    if task_id:
        task_tracker.update_progress(
            task_id, 
            TaskState.TRAINING_MODEL, 
            85, 
            "Training model: Final training with best parameters",
            {
                "stage": "final_training", 
                "best_params": best_params
            }
        )
    
    # Train final model with early stopping
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    final_model = xgb.XGBRegressor(
        **best_params,
        early_stopping_rounds=50
    )
    
    eval_set = [(X_train_final, y_train_final), (X_val, y_val)]
    final_model.fit(
        X_train_final,
        y_train_final,
        eval_set=eval_set,
        verbose=False
    )
    
    return final_model 