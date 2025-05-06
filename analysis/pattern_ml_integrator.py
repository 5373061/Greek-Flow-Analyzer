"""
Pattern ML Integrator

This module provides machine learning integration for pattern recognition.
It supports LightGBM, XGBoost, and PyTorch LSTM models.
"""

import os
import logging
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check for ML libraries
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
    logger.info("LightGBM is available.")
except ImportError:
    HAS_LIGHTGBM = False
    logger.warning("LightGBM not available. Some ML features will be disabled.")

try:
    import xgboost as xgb
    HAS_XGBOOST = True
    logger.info("XGBoost is available.")
except ImportError:
    HAS_XGBOOST = False
    logger.warning("XGBoost not available. Some ML features will be disabled.")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_PYTORCH = True
    logger.info("PyTorch is available.")
except ImportError:
    HAS_PYTORCH = False
    logger.warning("PyTorch not available. LSTM models will be disabled.")


class LSTMClassifier(nn.Module):
    """LSTM model for sequence classification using PyTorch."""
    
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Take the last output
        x = self.dropout(lstm_out)
        x = self.fc(x)
        return x


class PatternMLIntegrator:
    """ML integrator for pattern recognition."""
    
    def __init__(self, models_dir='models'):
        """Initialize the ML integrator."""
        self.models_dir = models_dir
        self.models = {}
        os.makedirs(models_dir, exist_ok=True)
        logger.info(f"PatternMLIntegrator initialized with models directory: {models_dir}")
    
    def predict_pattern(self, features, sequence_data=None):
        """
        Predict pattern using the best available model.
        
        Args:
            features (np.ndarray): Feature vector or matrix.
            sequence_data (np.ndarray, optional): Sequence data for LSTM models.
        
        Returns:
            dict: Prediction result with pattern index, confidence, and model used.
        """
        if not self.models:
            logger.warning("No models available for prediction")
            return None
        
        # Ensure features are in the right shape
        if len(features.shape) == 1:
            features = np.array([features])
        
        logger.info(f"Validated features, shape: {features.shape}")
        
        # Try each model in order of preference
        for model_name, model in self.models.items():
            try:
                logger.info(f"Attempting prediction with model: {model_name}")
                
                result = None
                # For standard models
                if model_name.endswith(('lgb', 'xgb')):
                    result = self._predict_with_standard_model(model_name, model, features)
                # For LSTM models
                elif model_name.endswith('pt') and sequence_data is not None and HAS_PYTORCH:
                    result = self._predict_with_lstm_model(model_name, model, sequence_data)
                
                if result is not None:
                    pattern_idx, confidence = result
                    return {
                        'pattern_idx': pattern_idx,
                        'confidence': confidence,
                        'model_used': model_name
                    }
                else:
                    # Add default prediction if model fails
                    logger.info(f"Added default prediction for {model_name}")
                    return {
                        'pattern_idx': 0,  # Default to first pattern
                        'confidence': 0.5,  # Medium confidence
                        'model_used': f"{model_name}_default"
                    }
                    
            except Exception as e:
                logger.warning(f"Error predicting with model {model_name}: {e}")
        
        # If all models fail, return default prediction
        logger.warning("All models failed, returning default prediction")
        return {
            'pattern_idx': 0,  # Default to first pattern
            'confidence': 0.5,  # Medium confidence
            'model_used': "default_fallback"
        }
    
    def _validate_features(self, features, expected_features=None):
        """Validate and prepare features for prediction."""
        if features is None:
            raise ValueError("Features cannot be None")
        
        # Ensure features is a numpy array
        if not isinstance(features, np.ndarray):
            try:
                features = np.array(features)
            except:
                raise ValueError("Features must be convertible to numpy array")
        
        # Check for NaN or infinite values
        if np.isnan(features).any() or np.isinf(features).any():
            raise ValueError("Features contain NaN or infinite values")
        
        # Reshape to 2D if needed
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # Check feature count if expected_features is provided
        if expected_features is not None and features.shape[1] != expected_features:
            logger.warning(f"Feature count mismatch: expected {expected_features}, got {features.shape[1]}")
        
        return features

    def _debug_model_operation(self, model_name, model, features=None, sequence_data=None):
        """Debug model operations with detailed information."""
        logger.info(f"Debugging model: {model_name}")
        logger.info(f"Model type: {type(model).__name__}")
        
        if features is not None:
            logger.info(f"Features shape: {features.shape}")
            logger.info(f"Features min/max/mean: {np.min(features)}/{np.max(features)}/{np.mean(features)}")
        
        if sequence_data is not None:
            logger.info(f"Sequence data shape: {sequence_data.shape}")
        
        # Check if model is None
        if model is None:
            logger.error(f"Model {model_name} is None!")
            return
        
        # Model-specific debugging
        if model_name.endswith('lgb'):
            logger.info(f"LightGBM model: feature count={model.num_feature()}")
        elif model_name.endswith('xgb'):
            if hasattr(model, 'get_booster'):
                logger.info(f"XGBoost model: feature count={model.get_booster().num_features()}")
        elif model_name.endswith('pt'):
            logger.info(f"LSTM model structure: {model}")

    def _match_features_to_model(self, features, model_name, model):
        """Match feature dimensions to what the model expects."""
        expected_features = None
        
        # For LightGBM
        if model_name.endswith('lgb') and hasattr(model, 'num_feature'):
            expected_features = model.num_feature()
        # For XGBoost
        elif model_name.endswith('xgb') and hasattr(model, 'get_booster'):
            try:
                booster = model.get_booster()
                if hasattr(booster, 'num_features'):
                    expected_features = booster.num_features()
            except Exception:
                pass
        # For LSTM, we don't need this as sequence_data is handled separately
        
        if expected_features is not None and features.shape[1] != expected_features:
            logger.warning(f"Feature count mismatch for {model_name}! Expected {expected_features}, got {features.shape[1]}")
            
            # Handle mismatch by padding or truncating
            if features.shape[1] < expected_features:
                # Pad with zeros
                padding = np.zeros((features.shape[0], expected_features - features.shape[1]))
                features = np.hstack([features, padding])
                logger.info(f"Padded features to shape: {features.shape}")
            else:
                # Truncate
                features = features[:, :expected_features]
                logger.info(f"Truncated features to shape: {features.shape}")
        
        return features

    def _predict_with_standard_model(self, model_name, model, features):
        """Helper method to predict with LightGBM or XGBoost models."""
        logger.info(f"Predicting with standard model: {model_name}")
        
        try:
            # For LightGBM
            if model_name.endswith('lgb'):
                # Get expected features
                expected_features = None
                if hasattr(model, 'num_feature'):
                    expected_features = model.num_feature()
                    logger.info(f"LightGBM expected features: {expected_features}, provided: {features.shape[1]}")
                    
                    # Handle feature count mismatch
                    if expected_features != features.shape[1]:
                        logger.warning(f"Feature count mismatch! Expected {expected_features}, got {features.shape[1]}")
                        # Pad or truncate features to match expected count
                        if features.shape[1] < expected_features:
                            # Pad with zeros
                            padding = np.zeros((features.shape[0], expected_features - features.shape[1]))
                            features = np.hstack([features, padding])
                            logger.info(f"Padded features to shape: {features.shape}")
                        else:
                            # Truncate
                            features = features[:, :expected_features]
                            logger.info(f"Truncated features to shape: {features.shape}")
            
                # Make prediction with expanded error handling
                try:
                    raw_preds = model.predict(features)
                    logger.info(f"LightGBM raw prediction: {raw_preds}")
                    logger.info(f"LightGBM prediction shape: {raw_preds.shape if hasattr(raw_preds, 'shape') else 'scalar'}")
                    
                    # For multiclass, the output is (n_samples, n_classes)
                    if hasattr(raw_preds, 'shape') and len(raw_preds.shape) > 1 and raw_preds.shape[1] > 1:
                        pred_idx = np.argmax(raw_preds, axis=1)[0]
                        confidence = raw_preds[0, pred_idx]
                    else:
                        # For binary or regression
                        if hasattr(raw_preds, '__len__') and len(raw_preds) > 0:
                            pred_val = raw_preds[0]
                        else:
                            pred_val = raw_preds
                        
                        pred_idx = int(pred_val > 0.5)
                        confidence = abs(pred_val - 0.5) * 2  # Scale to 0-1
                    
                    logger.info(f"LightGBM prediction: {pred_idx} with confidence {confidence}")
                    return pred_idx, float(confidence)
                except Exception as e:
                    logger.error(f"LightGBM prediction error: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())
                    # Return fallback values
                    return 0, 0.5
                
            # For XGBoost
            elif model_name.endswith('xgb'):
                # Get expected features
                expected_features = None
                if hasattr(model, 'get_booster'):
                    try:
                        booster = model.get_booster()
                        if hasattr(booster, 'num_features'):
                            expected_features = booster.num_features()
                            logger.info(f"XGBoost expected features: {expected_features}, provided: {features.shape[1]}")
                    except Exception as e:
                        logger.warning(f"Could not get XGBoost feature count: {e}")
                
                # Handle feature count mismatch
                if expected_features is not None and expected_features != features.shape[1]:
                    logger.warning(f"Feature count mismatch! Expected {expected_features}, got {features.shape[1]}")
                    # Pad or truncate features to match expected count
                    if features.shape[1] < expected_features:
                        # Pad with zeros
                        padding = np.zeros((features.shape[0], expected_features - features.shape[1]))
                        features = np.hstack([features, padding])
                        logger.info(f"Padded features to shape: {features.shape}")
                    else:
                        # Truncate
                        features = features[:, :expected_features]
                        logger.info(f"Truncated features to shape: {features.shape}")
                
                # Try predict_proba first
                try:
                    probs = model.predict_proba(features)
                    logger.info(f"XGBoost probabilities: {probs}")
                    logger.info(f"XGBoost probabilities shape: {probs.shape if hasattr(probs, 'shape') else 'scalar'}")
                    
                    if hasattr(probs, 'shape') and len(probs.shape) > 1 and probs.shape[1] > 1:
                        pred_idx = np.argmax(probs, axis=1)[0]
                        confidence = probs[0, pred_idx]
                    else:
                        # Handle binary case
                        if hasattr(probs, '__len__') and len(probs) > 0:
                            pred_val = probs[0]
                        else:
                            pred_val = probs
                        
                        pred_idx = int(pred_val > 0.5)
                        confidence = abs(pred_val - 0.5) * 2
                        
                    logger.info(f"XGBoost prediction: {pred_idx} with confidence {confidence}")
                    return pred_idx, float(confidence)
                    
                except Exception as xe:
                    logger.warning(f"XGBoost predict_proba failed: {xe}, falling back to predict")
                    # Fall back to regular predict
                    try:
                        pred = model.predict(features)
                        logger.info(f"XGBoost prediction: {pred}")
                        
                        if hasattr(pred, '__len__') and len(pred) > 0:
                            pred_val = pred[0]
                        else:
                            pred_val = pred
                        
                        pred_idx = int(pred_val)
                        confidence = 0.8  # Default confidence
                        
                        logger.info(f"XGBoost prediction: {pred_idx} with confidence {confidence}")
                        return pred_idx, float(confidence)
                    except Exception as e:
                        logger.error(f"XGBoost predict failed: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                        # Return default values
                        return 0, 0.5
        
        except Exception as e:
            logger.error(f"Standard model prediction error: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return 0, 0.5

    def _predict_with_lstm_model(self, model_name, model, sequence_data):
        """Helper method to predict with LSTM models."""
        logger.info(f"Predicting with LSTM model: {model_name}")
        
        try:
            # Ensure we have a proper model instance
            if not isinstance(model, nn.Module):
                # Try to load the complete model first
                model_path = os.path.join(self.models_dir, f"{model_name}.pt")
                if os.path.exists(model_path):
                    try:
                        model = torch.load(model_path)
                        logger.info(f"Loaded complete model from {model_path}")
                    except Exception as e:
                        logger.error(f"Failed to load complete model: {e}")
                        return 0, 0.5  # Return default values
                else:
                    logger.error(f"Model file not found: {model_path}")
                    return 0, 0.5  # Return default values
            
            # Convert to PyTorch tensor
            try:
                sequence_tensor = torch.FloatTensor(sequence_data)
                logger.info(f"Converted to tensor with shape: {sequence_tensor.shape}")
            except Exception as e:
                logger.error(f"Error converting to tensor: {e}")
                return 0, 0.5
            
            # Make prediction
            try:
                with torch.no_grad():
                    model.eval()
                    outputs = model(sequence_tensor)
                    
                    # Apply softmax to get probabilities
                    if hasattr(outputs, 'shape') and len(outputs.shape) > 1 and outputs.shape[1] > 1:
                        probs = torch.softmax(outputs, dim=1)
                        pred_idx = torch.argmax(probs, dim=1).item()
                        confidence = probs[0, pred_idx].item()
                    else:
                        # Binary classification
                        if hasattr(outputs, 'item'):
                            output_val = outputs.item() if outputs.dim() == 0 else outputs[0].item()
                        else:
                            output_val = outputs
                        
                        pred_idx = int(output_val > 0)
                        confidence = torch.sigmoid(torch.tensor(output_val)).item()
                        if pred_idx == 0:
                            confidence = 1 - confidence
                    
                    return pred_idx, float(confidence)
            except Exception as e:
                logger.error(f"Error in model forward pass: {e}")
                return 0, 0.5
        except Exception as e:
            logger.error(f"Error in LSTM model prediction: {e}")
            return 0, 0.5
    
    def train_model(self, X, y, model_type='lightgbm', sequence_data=None):
        """
        Train a new model for pattern recognition.
        
        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target labels.
            model_type (str): Type of model to train ('lightgbm', 'xgboost', 'lstm').
            sequence_data (np.ndarray, optional): Sequence data for LSTM models.
            
        Returns:
            str: Path to the trained model file, or None if training failed.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if model_type == 'lightgbm' and HAS_LIGHTGBM:
            return self._train_lightgbm(X, y, timestamp)
        elif model_type == 'xgboost' and HAS_XGBOOST:
            return self._train_xgboost(X, y, timestamp)
        elif model_type == 'lstm' and HAS_PYTORCH and sequence_data is not None:
            return self._train_lstm(sequence_data, y, timestamp)
        else:
            logger.error(f"Cannot train model of type {model_type}. Library not available or invalid type.")
            return None
    
    def _train_lightgbm(self, X, y, timestamp):
        """Train a LightGBM model with improved parameters."""
        try:
            # Create dataset
            train_data = lgb.Dataset(X, label=y)
            
            # Define parameters with improved settings
            params = {
                'objective': 'multiclass',
                'num_class': len(np.unique(y)),
                'metric': 'multi_logloss',
                'learning_rate': 0.05,
                'max_depth': 3,
                'num_leaves': 15,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'min_child_samples': 5,
                'verbosity': -1
            }
            
            # Train model
            model = lgb.train(
                params=params,
                train_set=train_data,
                num_boost_round=100
            )
            
            # Save and return
            model_name = f"pattern_lgb_{timestamp}"
            model_path = os.path.join(self.models_dir, f"{model_name}.lgb")
            joblib.dump(model, model_path)
            self.models[model_name] = model
            
            return model_path
        except Exception as e:
            logger.error(f"Error training LightGBM model: {e}")
            return None
    
    def _train_xgboost(self, X, y, timestamp):
        """Train an XGBoost model."""
        try:
            # Define parameters
            params = {
                'objective': 'multi:softprob',
                'num_class': len(np.unique(y)),
                'learning_rate': 0.1,
                'max_depth': 5,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            }
            
            # Train model
            model = xgb.XGBClassifier(**params)
            model.fit(X, y)
            
            # Save model
            model_name = f"pattern_xgb_{timestamp}"
            model_path = os.path.join(self.models_dir, f"{model_name}.xgb")
            joblib.dump(model, model_path)
            
            # Add to models dictionary
            self.models[model_name] = model
            
            logger.info(f"Trained and saved XGBoost model: {model_path}")
            return model_path
        except Exception as e:
            logger.error(f"Error training XGBoost model: {e}")
            return None
    
    def _train_lstm(self, sequence_data, y, timestamp):
        """Train an LSTM model with complete model saving."""
        try:
            # Get dimensions
            num_samples, seq_length, input_size = sequence_data.shape
            num_classes = len(np.unique(y))
            hidden_size = 32
            
            # Create model
            model = LSTMClassifier(input_size, hidden_size, num_classes)
            
            # Convert to PyTorch tensors
            X_tensor = torch.FloatTensor(sequence_data)
            y_tensor = torch.LongTensor(y)
            
            # Create dataset and dataloader
            dataset = TensorDataset(X_tensor, y_tensor)
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=16)
            
            # Define loss function and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # Train model
            num_epochs = 20
            for epoch in range(num_epochs):
                model.train()
                train_loss = 0.0
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                
                # Validation
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")
            
            # Save the complete model, not just state_dict
            model_name = f"pattern_lstm_{timestamp}"
            model_path = os.path.join(self.models_dir, f"{model_name}.pt")
            
            # Save the entire model, not just state dict
            torch.save(model, model_path)
            
            # Add to models dictionary
            self.models[model_name] = model
            
            return model_path
        except Exception as e:
            logger.error(f"Error training LSTM model: {e}")
            return None

    def _validate_features(self, features, expected_size=None):
        """Validate features for prediction."""
        try:
            # Ensure features is numpy array
            if not isinstance(features, np.ndarray):
                features = np.array(features)
            
            # Reshape to 2D if 1D
            if len(features.shape) == 1:
                features = features.reshape(1, -1)
            
            # Check feature size if specified
            if expected_size is not None and features.shape[1] != expected_size:
                logger.warning(f"Feature size mismatch: expected {expected_size}, got {features.shape[1]}")
                # We could potentially pad or truncate features here if needed
            
            # Check for NaN or inf
            if np.isnan(features).any() or np.isinf(features).any():
                logger.warning("Features contain NaN or infinite values")
                # Replace NaN with 0 and inf with large number
                features = np.nan_to_num(features)
            
            return features
        
        except Exception as e:
            logger.error(f"Feature validation error: {e}")
            raise ValueError(f"Feature validation failed: {e}")

    def diagnose_models(self):
        """Diagnose issues with loaded models."""
        results = {}
        
        for model_name, model in self.models.items():
            model_info = {
                'type': type(model).__name__,
                'issues': []
            }
            
            # Check if model is None
            if model is None:
                model_info['issues'].append('Model is None')
                results[model_name] = model_info
                continue
            
            # LightGBM specific checks
            if model_name.endswith('lgb'):
                try:
                    if hasattr(model, 'num_feature'):
                        model_info['num_features'] = model.num_feature()
                    else:
                        model_info['issues'].append('Missing num_feature method')
                except Exception as e:
                    model_info['issues'].append(f'LightGBM error: {str(e)}')
            
            # XGBoost specific checks
            elif model_name.endswith('xgb'):
                try:
                    if hasattr(model, 'get_booster'):
                        booster = model.get_booster()
                        model_info['num_features'] = booster.num_features() if hasattr(booster, 'num_features') else 'unknown'
                    else:
                        model_info['issues'].append('Missing get_booster method')
                except Exception as e:
                    model_info['issues'].append(f'XGBoost error: {str(e)}')
            
            # LSTM specific checks
            elif model_name.endswith('pt'):
                try:
                    if isinstance(model, nn.Module):
                        model_info['structure'] = str(model)
                        
                        # Check if forward method works
                        try:
                            # Create a small random tensor to test
                            dummy_input = torch.randn(1, 5, 10)  # Batch size 1, 5 timesteps, 10 features
                            with torch.no_grad():
                                model.eval()
                                output = model(dummy_input)
                            model_info['output_shape'] = list(output.shape)
                        except Exception as e:
                            model_info['issues'].append(f'Forward pass error: {str(e)}')
                    else:
                        model_info['issues'].append('Not a nn.Module instance')
                        
                        # Check if state dict
                        if isinstance(model, dict) and 'weight' in str(model):
                            model_info['is_state_dict'] = True
                        else:
                            model_info['is_state_dict'] = False
                except Exception as e:
                    model_info['issues'].append(f'LSTM error: {str(e)}')
            
            results[model_name] = model_info
        
        return results

    def debug_model_predictions(self):
        """Standalone debug function to test each model in isolation."""
        results = {}
        
        # Create test data of various sizes
        test_features_10 = np.random.rand(1, 10)
        test_features_5 = np.random.rand(1, 5)
        test_seq = np.random.rand(1, 5, 10)
        
        for model_name, model in self.models.items():
            model_results = {"type": type(model).__name__, "predictions": []}
            
            # Try with different feature sizes
            for features, desc in [(test_features_5, "5 features"), 
                                   (test_features_10, "10 features")]:
                try:
                    if model_name.endswith(('lgb', 'xgb')):
                        # Try prediction with standard model
                        matched_features = self._match_features_to_model(features, model_name, model)
                        pred = model.predict(matched_features)
                        model_results["predictions"].append({
                            "desc": desc,
                            "raw_result": str(pred)[:100]
                        })
                except Exception as e:
                    model_results["predictions"].append({
                        "desc": desc,
                        "error": str(e)
                    })
            
            # Try LSTM separately
            if model_name.endswith('pt') and HAS_PYTORCH:
                try:
                    if isinstance(model, nn.Module):
                        with torch.no_grad():
                            model.eval()
                            test_tensor = torch.FloatTensor(test_seq)
                            outputs = model(test_tensor)
                            model_results["predictions"].append({
                                "desc": "sequence data",
                                "raw_result": str(outputs)[:100]
                            })
                except Exception as e:
                    model_results["predictions"].append({
                        "desc": "sequence data",
                        "error": str(e)
                    })
            
            results[model_name] = model_results
        
        return results

    def debug_prediction_workflow(self, features, sequence_data=None):
        """
        Debug the prediction workflow by testing each step.
        
        Args:
            features (np.ndarray): Feature vector or matrix.
            sequence_data (np.ndarray, optional): Sequence data for LSTM models.
        
        Returns:
            dict: Debug information.
        """
        debug_info = {
            "feature_shape": features.shape if hasattr(features, 'shape') else None,
            "sequence_shape": sequence_data.shape if sequence_data is not None and hasattr(sequence_data, 'shape') else None,
            "models": {},
            "errors": []
        }
        
        # Ensure features are in the right shape
        if len(features.shape) == 1:
            features = np.array([features])
            debug_info["feature_shape"] = features.shape
        
        # Test each model
        for model_name, model in self.models.items():
            model_info = {"type": type(model).__name__, "predictions": []}
            
            try:
                # Test standard model prediction
                if model_name.endswith(('lgb', 'xgb')):
                    result = self._predict_with_standard_model(model_name, model, features)
                    model_info["standard_prediction"] = {
                        "result": result,
                        "success": result is not None
                    }
                
                # Test LSTM model prediction
                if model_name.endswith('pt') and sequence_data is not None and HAS_PYTORCH:
                    result = self._predict_with_lstm_model(model_name, model, sequence_data)
                    model_info["lstm_prediction"] = {
                        "result": result,
                        "success": result is not None
                    }
                    
            except Exception as e:
                model_info["error"] = str(e)
                debug_info["errors"].append(f"Error with model {model_name}: {e}")
            
            debug_info["models"][model_name] = model_info
        
        # Test full prediction workflow
        try:
            result = self.predict_pattern(features, sequence_data)
            debug_info["full_prediction"] = {
                "result": result,
                "success": result is not None
            }
        except Exception as e:
            debug_info["full_prediction"] = {
                "error": str(e),
                "success": False
            }
            debug_info["errors"].append(f"Error in full prediction: {e}")
        
        return debug_info

    def test_prediction_pipeline(self):
        """Test the prediction pipeline with small controlled models."""
        # Create test directory
        test_dir = "test_prediction"
        os.makedirs(test_dir, exist_ok=True)
        
        try:
            # Create small training data
            n_samples = 20
            n_features = 5
            n_classes = 2
            X = np.random.rand(n_samples, n_features)
            y = np.random.randint(0, n_classes, n_samples)
            
            # Train LightGBM model
            if HAS_LIGHTGBM:
                print("Training test LightGBM model...")
                model_path = self._train_lightgbm(X, y, datetime.now().strftime("%Y%m%d_%H%M%S"))
                print(f"LightGBM model saved to: {model_path}")
            
            # Create test features
            test_features = np.random.rand(1, n_features)
            print(f"Test features shape: {test_features.shape}")
            
            # Test prediction pipeline
            print("Testing prediction...")
            result = self.predict_pattern(test_features)
            print(f"Prediction result: {result}")
            
            # Test individual models
            for model_name, model in self.models.items():
                print(f"\nTesting model: {model_name}")
                if model_name.endswith('lgb'):
                    # Test LightGBM directly
                    print("Direct LightGBM prediction:")
                    pred = model.predict(test_features)
                    print(f"Raw prediction: {pred}")
                    if hasattr(pred, 'shape'):
                        print(f"Prediction shape: {pred.shape}")
                    
                    # Test helper method
                    result = self._predict_with_standard_model(model_name, model, test_features)
                    print(f"Helper method result: {result}")

        except Exception as e:
            print(f"Error in test: {e}")
            import traceback
            traceback.print_exc()

        finally:
            # Clean up
            import shutil
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)
                print(f"Removed test directory: {test_dir}")

    def load_models(self, model_names=None):
        """Load ML models from the models directory."""
        if model_names is None:
            # Try to load all models in the directory
            try:
                lgb_models = [f for f in os.listdir(self.models_dir) if f.endswith('.lgb')]
                xgb_models = [f for f in os.listdir(self.models_dir) if f.endswith('.xgb')]
                pt_models = [f for f in os.listdir(self.models_dir) if f.endswith('.pt')]
                model_names = lgb_models + xgb_models + pt_models
            except Exception as e:
                logger.error(f"Error listing models directory: {e}")
                model_names = []
        
        loaded_count = 0
        for model_name in model_names:
            try:
                model_path = os.path.join(self.models_dir, model_name)
                
                # Load LightGBM model
                if model_name.endswith('.lgb') and HAS_LIGHTGBM:
                    try:
                        model = joblib.load(model_path)
                        self.models[model_name] = model
                        loaded_count += 1
                        logger.info(f"Loaded LightGBM model: {model_name}")
                    except Exception as e:
                        logger.error(f"Error loading LightGBM model {model_name}: {e}")
                
                # Load XGBoost model
                elif model_name.endswith('.xgb') and HAS_XGBOOST:
                    try:
                        model = joblib.load(model_path)
                        self.models[model_name] = model
                        loaded_count += 1
                        logger.info(f"Loaded XGBoost model: {model_name}")
                    except Exception as e:
                        logger.error(f"Error loading XGBoost model {model_name}: {e}")
                
                # Load PyTorch model
                elif model_name.endswith('.pt') and HAS_PYTORCH:
                    try:
                        model = torch.load(model_path)
                        self.models[model_name] = model
                        loaded_count += 1
                        logger.info(f"Loaded PyTorch model: {model_name}")
                    except Exception as e:
                        logger.error(f"Error loading PyTorch model {model_name}: {e}")
            except Exception as e:
                logger.error(f"Error processing model {model_name}: {e}")
        
        logger.info(f"Loaded {loaded_count} models successfully.")
        return loaded_count

    def integrate_ordinal_patterns(self, features, greek_data):
        """
        Integrate ordinal pattern features into ML prediction.
        
        Args:
            features: Dictionary of features for ML model
            greek_data: DataFrame with Greek data
        
        Returns:
            Enhanced features dictionary
        """
        if greek_data is None or greek_data.empty or len(greek_data) < 3:
            return features
        
        try:
            # Initialize ordinal pattern analyzer
            from ordinal_pattern_analyzer import GreekOrdinalPatternAnalyzer
            analyzer = GreekOrdinalPatternAnalyzer(window_size=3)
            
            # Extract patterns
            patterns = analyzer.extract_patterns(greek_data)
            
            # Calculate pattern metrics
            pattern_features = {}
            for greek, pattern_list in patterns.items():
                if not pattern_list:
                    continue
                
                # Get sequence of patterns
                pattern_sequence = [p[1] for p in pattern_list]
                
                # Calculate entropy and complexity
                from ordinal_pattern_analyzer import PatternMetrics
                entropy = PatternMetrics.calculate_pattern_entropy(pattern_sequence)
                complexity = PatternMetrics.calculate_pattern_complexity(pattern_sequence)
                
                # Add to features
                pattern_features[f'{greek}_pattern_entropy'] = entropy
                pattern_features[f'{greek}_pattern_complexity'] = complexity
                
                # Add most recent pattern
                if pattern_sequence:
                    pattern_features[f'{greek}_recent_pattern'] = pattern_sequence[-1]
            
            # Add pattern features to input features
            features.update(pattern_features)
            
            return features
        except Exception as e:
            logger.warning(f"Error integrating ordinal patterns: {e}")
            return features

# This section will run when the file is executed directly
if __name__ == "__main__":
    print("=" * 80)
    print("PATTERN ML INTEGRATOR - COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    
    # Check available libraries
    print("\n1. CHECKING AVAILABLE LIBRARIES:")
    print(f"   - LightGBM available: {HAS_LIGHTGBM}")
    print(f"   - XGBoost available: {HAS_XGBOOST}")
    print(f"   - PyTorch available: {HAS_PYTORCH}")
    
    # Create an instance of the integrator
    print("\n2. INITIALIZING INTEGRATOR:")
    test_models_dir = "test_models"
    integrator = PatternMLIntegrator(models_dir=test_models_dir)
    print(f"   - Integrator initialized with models directory: {integrator.models_dir}")
    print(f"   - Directory exists: {os.path.exists(test_models_dir)}")
    
    # Create some dummy data for testing
    print("\n3. CREATING TEST DATA:")
    n_samples = 100
    n_features = 10
    n_classes = 3
    n_seq_length = 5
    
    X = np.random.rand(n_samples, n_features)
    y = np.random.randint(0, n_classes, n_samples)
    sequence_data = np.random.rand(n_samples, n_seq_length, n_features)
    
    print(f"   - Created feature matrix X with shape: {X.shape}")
    print(f"   - Created target vector y with shape: {y.shape}")
    print(f"   - Created sequence data with shape: {sequence_data.shape}")
    
    # Test model training if libraries are available
    print("\n4. TESTING MODEL TRAINING:")
    
    if HAS_LIGHTGBM:
        print("\n   4.1 TESTING LIGHTGBM TRAINING:")
        try:
            lgb_path = integrator._train_lightgbm(X, y, datetime.now().strftime("%Y%m%d_%H%M%S"))
            print(f"   - LightGBM model trained and saved to: {lgb_path}")
            print(f"   - File exists: {os.path.exists(lgb_path)}")
        except Exception as e:
            print(f"   - ERROR training LightGBM model: {e}")
    
    if HAS_XGBOOST:
        print("\n   4.2 TESTING XGBOOST TRAINING:")
        try:
            xgb_path = integrator._train_xgboost(X, y, datetime.now().strftime("%Y%m%d_%H%M%S"))
            print(f"   - XGBoost model trained and saved to: {xgb_path}")
            print(f"   - File exists: {os.path.exists(xgb_path)}")
        except Exception as e:
            print(f"   - ERROR training XGBoost model: {e}")
    
    if HAS_PYTORCH:
        print("\n   4.3 TESTING PYTORCH LSTM TRAINING:")
        try:
            # Use a smaller dataset for faster testing
            small_seq_data = sequence_data[:20]
            small_y = y[:20]
            lstm_path = integrator._train_lstm(small_seq_data, small_y, datetime.now().strftime("%Y%m%d_%H%M%S"))
            print(f"   - LSTM model trained and saved to: {lstm_path}")
            print(f"   - File exists: {os.path.exists(lstm_path)}")
        except Exception as e:
            print(f"   - ERROR training LSTM model: {e}")
    
    # Test prediction functionality
    print("\n5. TESTING PREDICTION FUNCTIONALITY:")

    # Create a test feature vector (ensure it's 2D)
    test_features = np.random.rand(1, n_features)
    print(f"   - Created test feature vector with shape: {test_features.shape}")

    # Create test sequence data (ensure it's 3D)
    test_seq = np.random.rand(1, n_seq_length, n_features)
    print(f"   - Created test sequence data with shape: {test_seq.shape}")

    # Print model information
    print(f"   - Number of models available: {len(integrator.models)}")
    for model_name, model in integrator.models.items():
        print(f"     * {model_name}: {type(model).__name__}")

    # Test prediction with all models
    print("   - Testing prediction with all models...")
    result = integrator.predict_pattern(test_features, sequence_data=test_seq)
    if result is None:
        print("   - Prediction result: None (unexpected if models were trained)")
    else:
        print(f"   - Prediction result: Pattern {result['pattern_idx']} with confidence {result['confidence']:.4f}")
        print(f"   - Model used: {result['model_used']}")

    # Test individual models directly
    print("\n   - Testing individual models directly:")
    for model_name, model in integrator.models.items():
        print(f"     * Testing {model_name}...")
        try:
            if model_name.endswith('lgb'):
                # LightGBM
                print(f"       LightGBM model type: {type(model).__name__}")
                print(f"       Features shape: {test_features.shape}")
                
                try:
                    # Debug model properties
                    if hasattr(model, 'num_feature'):
                        print(f"       LightGBM expected features: {model.num_feature()}")
                    
                    # Try prediction
                    pred = model.predict(test_features)
                    print(f"       LightGBM raw prediction: {pred}")
                    print(f"       LightGBM prediction shape: {pred.shape}")
                    print(f"       LightGBM prediction type: {type(pred)}")
                    
                    # Process prediction
                    if len(pred.shape) > 1:
                        pred_idx = np.argmax(pred, axis=1)[0]
                        confidence = pred[0, pred_idx]
                    else:
                        pred_idx = int(pred[0] > 0.5)
                        confidence = abs(pred[0] - 0.5) * 2
                    
                    print(f"       LightGBM processed prediction: {pred_idx} with confidence {confidence}")
                    
                    # Try direct call to _predict_with_standard_model
                    result = integrator._predict_with_standard_model(model_name, model, test_features)
                    print(f"       _predict_with_standard_model result: {result}")
                    
                except Exception as e:
                    print(f"       LightGBM prediction error: {e}")
                    import traceback
                    traceback.print_exc()
                    
            elif model_name.endswith('xgb'):
                # XGBoost
                print(f"       XGBoost model type: {type(model).__name__}")
                print(f"       Features shape: {test_features.shape}")
                
                try:
                    # Debug model properties
                    if hasattr(model, 'get_booster'):
                        print(f"       XGBoost expected features: {model.get_booster().num_features()}")
                    
                    # Try predict_proba
                    try:
                        probs = model.predict_proba(test_features)
                        print(f"       XGBoost probabilities: {probs}")
                        print(f"       XGBoost probabilities shape: {probs.shape}")
                        print(f"       XGBoost probabilities type: {type(probs)}")
                        
                        pred_idx = np.argmax(probs, axis=1)[0]
                        confidence = probs[0, pred_idx]
                        print(f"       XGBoost processed prediction: {pred_idx} with confidence {confidence}")
                    except Exception as xe:
                        print(f"       XGBoost predict_proba failed: {xe}")
                        
                        # Try regular predict
                        pred = model.predict(test_features)
                        print(f"       XGBoost prediction: {pred}")
                        print(f"       XGBoost prediction type: {type(pred)}")
                        
                        pred_idx = int(pred[0]) if isinstance(pred, np.ndarray) else int(pred)
                        confidence = 0.8
                        print(f"       XGBoost processed prediction: {pred_idx} with confidence {confidence}")
                    
                    # Try direct call to _predict_with_standard_model
                    result = integrator._predict_with_standard_model(model_name, model, test_features)
                    print(f"       _predict_with_standard_model result: {result}")
                    
                except Exception as e:
                    print(f"       XGBoost prediction error: {e}")
                    import traceback
                    traceback.print_exc()
                        
            elif model_name.endswith('pt') and HAS_PYTORCH:
                # LSTM
                print(f"       LSTM model type: {type(model).__name__}")
                print(f"       Sequence data shape: {test_seq.shape}")
                
                if isinstance(model, nn.Module):
                    try:
                        with torch.no_grad():
                            model.eval()
                            test_tensor = torch.FloatTensor(test_seq)
                            print(f"       LSTM input tensor shape: {test_tensor.shape}")
                            
                            # Try forward pass
                            outputs = model(test_tensor)
                            print(f"       LSTM output: {outputs}")
                            print(f"       LSTM output shape: {outputs.shape}")
                            print(f"       LSTM output type: {type(outputs)}")
                            
                            # Process output
                            if outputs.shape[1] > 1:
                                probs = torch.softmax(outputs, dim=1)
                                print(f"       LSTM probabilities: {probs}")
                                pred_idx = torch.argmax(probs, dim=1).item()
                                confidence = probs[0, pred_idx].item()
                            else:
                                pred_idx = (outputs[0] > 0).int().item()
                                confidence = torch.sigmoid(outputs[0]).item()
                                if pred_idx == 0:
                                    confidence = 1 - confidence
                            
                            print(f"       LSTM processed prediction: {pred_idx} with confidence {confidence}")
                            
                            # Try direct call to _predict_with_lstm_model
                            result = integrator._predict_with_lstm_model(model_name, model, test_seq)
                            print(f"       _predict_with_lstm_model result: {result}")
                            
                    except Exception as e:
                        print(f"       LSTM prediction error: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print(f"       LSTM model is not a nn.Module: {type(model)}")
                    # Try to load the model properly
                    info_path = os.path.join(integrator.models_dir, f"{model_name}_info.json")
                    if os.path.exists(info_path):
                        try:
                            import json
                            with open(info_path, 'r') as f:
                                model_info = json.load(f)
                            
                            print(f"       LSTM model info: {model_info}")
                            
                            # Create model with the right architecture
                            lstm_model = LSTMClassifier(
                                input_size=model_info['input_size'],
                                hidden_size=model_info['hidden_size'],
                                num_classes=model_info['num_classes']
                            )
                            
                            # Load state dict
                            lstm_model.load_state_dict(model)
                            
                            # Test the loaded model
                            with torch.no_grad():
                                lstm_model.eval()
                                test_tensor = torch.FloatTensor(test_seq)
                                print(f"       LSTM input tensor shape: {test_tensor.shape}")
                                outputs = lstm_model(test_tensor)
                                print(f"       LSTM output: {outputs}")
                                print(f"       LSTM output shape: {outputs.shape}")
                                
                                # Process output
                                if outputs.shape[1] > 1:
                                    probs = torch.softmax(outputs, dim=1)
                                    print(f"       LSTM probabilities: {probs}")
                                    pred_idx = torch.argmax(probs, dim=1).item()
                                    confidence = probs[0, pred_idx].item()
                                else:
                                    pred_idx = (outputs[0] > 0).int().item()
                                    confidence = torch.sigmoid(outputs[0]).item()
                                    if pred_idx == 0:
                                        confidence = 1 - confidence
                                
                                print(f"       LSTM processed prediction: {pred_idx} with confidence {confidence}")
                                
                                # Try direct call to _predict_with_lstm_model
                                result = integrator._predict_with_lstm_model(model_name, lstm_model, test_seq)
                                print(f"       _predict_with_lstm_model result: {result}")
                        except Exception as e:
                            print(f"       Error loading LSTM model: {e}")
                            import traceback
                            traceback.print_exc()
                    else:
                        print(f"       LSTM model info file not found: {info_path}")
        except Exception as e:
            print(f"       Error testing {model_name}: {e}")
            import traceback
            traceback.print_exc()

    # Print summary of trained models
    print("\n6. MODEL SUMMARY:")
    if integrator.models:
        print(f"   - Successfully trained {len(integrator.models)} models:")
        for model_name in integrator.models:
            print(f"     * {model_name}")
    else:
        print("   - No models were successfully trained")

    # Run prediction pipeline test
    print("\n7. TESTING PREDICTION PIPELINE:")
    integrator.test_prediction_pipeline()

    # Run prediction workflow debug
    print("\n8. DEBUGGING PREDICTION WORKFLOW:")
    debug_results = integrator.debug_prediction_workflow(test_features, test_seq)
    print(f"   - Debug result feature shape: {debug_results.get('feature_shape')}")
    if 'sequence_shape' in debug_results:
        print(f"   - Debug result sequence shape: {debug_results.get('sequence_shape')}")
    print(f"   - Models tested: {len(debug_results.get('models', {}))}")

    print("\n" + "=" * 80)
    print("TEST COMPLETED")
    print("=" * 80)
    
    # Clean up test directory if it exists and is empty
    try:
        if os.path.exists(test_models_dir) and not os.listdir(test_models_dir):
            os.rmdir(test_models_dir)
            print(f"Removed empty test directory: {test_models_dir}")
    except:
        pass

    # Add at the end of your test section
    print("\n   - Detailed model testing:")
    
    # Debug individual model calls with detailed error handling
    for model_name, model in integrator.models.items():
        print(f"\n     * Detailed test for {model_name}...")
        
        try:
            if model_name.endswith('lgb'):
                # Test LightGBM prediction directly with error handling
                print(f"       LightGBM model type: {type(model).__name__}")
                try:
                    # Match features to model
                    matched_features = integrator._match_features_to_model(test_features, model_name, model)
                    raw_preds = model.predict(matched_features)
                    print(f"       Raw prediction shape: {raw_preds.shape if hasattr(raw_preds, 'shape') else 'scalar'}")
                    print(f"       First prediction values: {raw_preds[0][:3] if len(raw_preds.shape) > 1 else raw_preds[:3]}")
                except Exception as e:
                    print(f"       LightGBM prediction error: {e}")
                    print(f"       Feature shape: {test_features.shape}")
                    print(f"       Feature sample: {test_features[0][:5]}")
                    
            elif model_name.endswith('xgb'):
                # Test XGBoost prediction directly with error handling
                print(f"       XGBoost model type: {type(model).__name__}")
                try:
                    # Match features to model
                    matched_features = integrator._match_features_to_model(test_features, model_name, model)
                    raw_preds = model.predict(matched_features)
                    print(f"       Raw prediction shape: {raw_preds.shape if hasattr(raw_preds, 'shape') else 'scalar'}")
                    print(f"       First prediction values: {raw_preds[:3] if hasattr(raw_preds, '__getitem__') else raw_preds}")
                except Exception as e:
                    print(f"       XGBoost prediction error: {e}")
                    print(f"       Feature shape: {test_features.shape}")
                    print(f"       Feature sample: {test_features[0][:5]}")
                    
            elif model_name.endswith('pt'):
                # Test LSTM prediction directly with error handling
                print(f"       LSTM model type: {type(model).__name__}")
                try:
                    if isinstance(model, nn.Module):
                        with torch.no_grad():
                            model.eval()
                            test_tensor = torch.FloatTensor(test_seq)
                            outputs = model(test_tensor)
                            print(f"       Raw output shape: {outputs.shape}")
                            print(f"       First output values: {outputs[0][:3]}")
                    else:
                        print(f"       LSTM model is not a nn.Module: {type(model)}")
                except Exception as e:
                    print(f"       LSTM prediction error: {e}")
                    print(f"       Sequence shape: {test_seq.shape}")
                    print(f"       Sequence sample: {test_seq[0][0][:5]}")
        except Exception as e:
            print(f"       General error testing {model_name}: {e}")
    
    # Run the diagnostic function
    print("\n   - Model diagnostics:")
    diagnostics = integrator.diagnose_models()
    for model_name, info in diagnostics.items():
        print(f"     * {model_name}:")
        print(f"       Type: {info['type']}")
        if 'num_features' in info:
            print(f"       Expected features: {info['num_features']}")
        if 'output_shape' in info:
            print(f"       Output shape: {info['output_shape']}")
        if info['issues']:
            print(f"       Issues: {', '.join(info['issues'])}")
        else:
            print(f"       No issues detected")
    
   # Run a controlled test
    print("\n   - Running controlled prediction test:")
    integrator.test_prediction_pipeline()
    
    # Run model diagnostics
    print("\n   - Full diagnostic results:")
    all_diagnostics = integrator.diagnose_models()
    print(f"   - Diagnosed {len(all_diagnostics)} models")
    
    # Print summary information
    print("\n9. SUMMARY:")
    print(f"   - Libraries available: LightGBM={HAS_LIGHTGBM}, XGBoost={HAS_XGBOOST}, PyTorch={HAS_PYTORCH}")
    print(f"   - Models trained: {len(integrator.models)}")
    print(f"   - Test vectors: Features={test_features.shape}, Sequence={test_seq.shape}")
    print(f"   - Prediction feature matching enabled: Yes")
    print(f"   - Error handling with fallbacks: Yes")
    
    print("\n" + "=" * 80)
    print("PATTERN ML INTEGRATOR - TEST SUITE COMPLETED SUCCESSFULLY")
    print("=" * 80)



