#!/usr/bin/env python3
# pattern_analyzer.py - Pattern recognition for options analysis

import logging
import numpy as np
import pandas as pd
from datetime import datetime
import os
import json
import joblib

# Add import for PatternMLIntegrator
try:
    from analysis.pattern_ml_integrator import PatternMLIntegrator
    HAS_ML_INTEGRATOR = True
except ImportError:
    HAS_ML_INTEGRATOR = False

# Configure logging
logger = logging.getLogger(__name__)

class PatternRecognizer:
    """
    Recognizes market patterns based on Greek energy flow and momentum data.
    """
    
    def __init__(self, model_path=None, models_dir='models'):
        """
        Initialize the pattern recognizer.
        
        Args:
            model_path (str, optional): Path to a trained model file. If None, uses rule-based approach.
            models_dir (str): Directory containing trained models.
        """
        self.model_path = model_path
        self.models_dir = models_dir
        
        # Load patterns from JSON file if available, otherwise use defaults
        self.patterns = self._load_patterns()
        
        # Initialize ML integrator if available
        self.ml_integrator = None
        if HAS_ML_INTEGRATOR:
            try:
                self.ml_integrator = PatternMLIntegrator(models_dir=models_dir)
                logger.info(f"ML integrator initialized with models directory: {models_dir}")
                
                # Load specific model if provided
                if model_path and os.path.exists(model_path):
                    model_name = os.path.basename(model_path).split('.')[0]
                    model_ext = os.path.basename(model_path).split('.')[-1]
                    
                    if model_ext == 'lgb':
                        self.ml_integrator.models[model_name] = joblib.load(model_path)
                        logger.info(f"Loaded LightGBM model: {model_name}")
                    elif model_ext == 'xgb':
                        self.ml_integrator.models[model_name] = joblib.load(model_path)
                        logger.info(f"Loaded XGBoost model: {model_name}")
                    elif model_ext == 'pt':
                        try:
                            import torch
                            self.ml_integrator.models[model_name] = torch.load(model_path)
                            logger.info(f"Loaded PyTorch model: {model_name}")
                        except ImportError:
                            logger.warning("PyTorch not available, cannot load LSTM model")
                    else:
                        logger.warning(f"Unknown model type: {model_ext}")
            except Exception as e:
                logger.warning(f"Failed to initialize ML integrator: {e}")
    
    def _load_patterns(self):
        """Load pattern definitions from JSON file or use defaults."""
        try:
            patterns_file = os.path.join(os.path.dirname(__file__), 'patterns.json')
            if os.path.exists(patterns_file):
                with open(patterns_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Error loading patterns file: {e}")
        
        # Default patterns if file not found or error occurs
        return {
            "BULLISH_MOMENTUM": {
                "description": "Strong bullish momentum with positive energy flow",
                "conditions": {
                    "market_regime": ["BULLISH", "NEUTRAL"],
                    "energy_direction": ["UP", "ACCELERATING"],
                    "anomalies": ["GAMMA_SPIKE", "VANNA_SURGE"]
                }
            },
            "BEARISH_MOMENTUM": {
                "description": "Strong bearish momentum with negative energy flow",
                "conditions": {
                    "market_regime": ["BEARISH", "NEUTRAL"],
                    "energy_direction": ["DOWN", "ACCELERATING"],
                    "anomalies": ["GAMMA_SPIKE", "VANNA_COLLAPSE"]
                }
            },
            "CONSOLIDATION": {
                "description": "Price consolidation with decreasing volatility",
                "conditions": {
                    "market_regime": ["NEUTRAL"],
                    "energy_direction": ["FLAT", "DECELERATING"],
                    "anomalies": ["VOL_COMPRESSION"]
                }
            },
            "BREAKOUT_IMMINENT": {
                "description": "Potential breakout with energy building up",
                "conditions": {
                    "market_regime": ["NEUTRAL"],
                    "energy_direction": ["BUILDING", "FLAT"],
                    "anomalies": ["GAMMA_WALL", "CHARM_ACCELERATION"]
                }
            },
            "REVERSAL_SIGNAL": {
                "description": "Potential trend reversal with energy shift",
                "conditions": {
                    "market_regime": ["BULLISH", "BEARISH"],
                    "energy_direction": ["REVERSING", "SHIFTING"],
                    "anomalies": ["DELTA_FLIP", "GAMMA_REVERSAL"]
                }
            },
            "COMPLEX": {
                "description": "Complex pattern with mixed signals",
                "conditions": {}
            }
        }
    
    def predict_pattern(self, greek_results, momentum_data, use_ml=True, sequence_data=None):
        """
        Predict the market pattern based on Greek results and momentum data.
        
        Args:
            greek_results (dict): Results from Greek energy flow analysis.
            momentum_data (dict): Momentum indicators.
            use_ml (bool): Whether to use ML models if available.
            sequence_data (np.ndarray, optional): Sequence data for LSTM models.
        
        Returns:
            dict: Predicted pattern with confidence and description.
        """
        # Handle missing data
        if greek_results is None or momentum_data is None:
            return {
                "pattern": "COMPLEX",
                "confidence": 0.6,
                "description": self.patterns["COMPLEX"]["description"]
            }
        
        # If ML integrator is available and use_ml is True, try ML prediction first
        if use_ml and self.ml_integrator is not None:
            try:
                # Extract features for ML model
                features = self._extract_features(greek_results, momentum_data)
                
                # Get prediction from ML integrator
                ml_result = self.ml_integrator.predict_pattern(features, sequence_data=sequence_data)
                
                if ml_result is not None:
                    # Map pattern index to pattern name
                    pattern_names = list(self.patterns.keys())
                    pattern_idx = ml_result['pattern_idx']
                    pattern_name = pattern_names[pattern_idx] if pattern_idx < len(pattern_names) else "COMPLEX"
                    
                    logger.info(f"ML prediction: {pattern_name} (Confidence: {ml_result['confidence']:.2f}, Model: {ml_result.get('model_used', 'unknown')})")
                    
                    return {
                        "pattern": pattern_name,
                        "confidence": ml_result['confidence'],
                        "description": self.patterns[pattern_name]["description"],
                        "model_used": ml_result.get('model_used', 'unknown')
                    }
            except Exception as e:
                logger.warning(f"ML prediction failed: {e}. Falling back to rule-based approach.")
        
        # Fall back to rule-based approach
        return self._predict_with_rules(greek_results, momentum_data)
    
    def predict_pattern_ensemble(self, greek_results, momentum_data, sequence_data=None):
        """
        Predict pattern using ensemble of all available models.
        
        Args:
            greek_results (dict): Results from Greek energy flow analysis.
            momentum_data (dict): Momentum indicators.
            sequence_data (np.ndarray, optional): Sequence data for LSTM models.
        
        Returns:
            dict: Predicted pattern with confidence and description.
        """
        if not HAS_ML_INTEGRATOR or self.ml_integrator is None or not self.ml_integrator.models:
            # Fall back to rule-based if no ML models available
            result = self._predict_with_rules(greek_results, momentum_data)
            # Add empty ensemble info to maintain consistent return structure
            result["ensemble_info"] = {
                "models_used": 0,
                "supporting_models": [],
                "vote_distribution": {}
            }
            return result
        
        try:
            # Extract features
            features = self._extract_features(greek_results, momentum_data)
            features_array = np.array([features])
            
            # Get predictions from all models
            predictions = []
            pattern_names = list(self.patterns.keys())
            
            for model_name, model in self.ml_integrator.models.items():
                try:
                    # Get prediction based on model type
                    if model_name.endswith('pt') and sequence_data is not None:
                        pred = self.ml_integrator._predict_with_lstm_model(model_name, model, sequence_data)
                    else:
                        pred = self.ml_integrator._predict_with_standard_model(model_name, model, features_array)
                    
                    if pred is not None:
                        pattern_idx, confidence = pred
                        predictions.append({
                            'pattern_idx': pattern_idx,
                            'confidence': confidence,
                            'model_name': model_name
                        })
                except Exception as e:
                    logger.warning(f"Error getting prediction from model {model_name}: {e}")
            
            if not predictions:
                # Fall back to rule-based if all models failed
                result = self._predict_with_rules(greek_results, momentum_data)
                # Add empty ensemble info to maintain consistent return structure
                result["ensemble_info"] = {
                    "models_used": 0,
                    "supporting_models": [],
                    "vote_distribution": {}
                }
                return result
            
            # Weighted voting based on confidence
            pattern_votes = {}
            for pred in predictions:
                pattern_idx = pred['pattern_idx']
                confidence = pred['confidence']
                
                if pattern_idx not in pattern_votes:
                    pattern_votes[pattern_idx] = 0
                
                pattern_votes[pattern_idx] += confidence
            
            # Find pattern with highest vote
            if not pattern_votes:
                # Fallback if voting somehow failed
                pattern_idx = 0
                avg_confidence = 0.5
            else:
                pattern_idx = max(pattern_votes, key=pattern_votes.get)
                # Calculate average confidence for the winning pattern
                supporting_preds = [p for p in predictions if p['pattern_idx'] == pattern_idx]
                avg_confidence = sum(p['confidence'] for p in supporting_preds) / len(supporting_preds)
            
            # Get pattern name
            if pattern_idx < len(pattern_names):
                pattern_name = pattern_names[pattern_idx]
            else:
                pattern_name = "COMPLEX"
            
            # Get supporting models
            supporting_models = [p['model_name'] for p in predictions if p['pattern_idx'] == pattern_idx]
            
            logger.info(f"Ensemble prediction: {pattern_name} with confidence {avg_confidence:.2f}")
            logger.info(f"Models used: {len(predictions)}")
            
            return {
                "pattern": pattern_name,
                "confidence": avg_confidence,
                "description": self.patterns[pattern_name]["description"],
                "ensemble_info": {
                    "models_used": len(predictions),
                    "supporting_models": supporting_models,
                    "vote_distribution": pattern_votes
                }
            }
            
        except Exception as e:
            logger.error(f"Error in ensemble prediction: {e}")
            result = self._predict_with_rules(greek_results, momentum_data)
            # Add empty ensemble info to maintain consistent return structure
            result["ensemble_info"] = {
                "models_used": 0,
                "supporting_models": [],
                "vote_distribution": {}
            }
            return result
    
    def _extract_features(self, greek_results, momentum_data):
        """
        Extract and normalize features for ML model from Greek results and momentum data.
        
        Args:
            greek_results (dict): Results from Greek energy flow analysis.
            momentum_data (dict): Momentum indicators.
            
        Returns:
            list: Feature vector for ML model.
        """
        features = []
        
        # Extract features from greek_results
        delta_exposure = greek_results.get('delta_exposure', 0)
        gamma_exposure = greek_results.get('gamma_exposure', 0)
        vanna_exposure = greek_results.get('vanna_exposure', 0)
        charm_exposure = greek_results.get('charm_exposure', 0)
        
        # Normalize exposures to handle different scales
        max_exposure = max(abs(delta_exposure), abs(gamma_exposure), 
                           abs(vanna_exposure), abs(charm_exposure), 1e-6)
        
        # Add normalized exposures
        features.append(float(delta_exposure / max_exposure))
        features.append(float(gamma_exposure / max_exposure))
        features.append(float(vanna_exposure / max_exposure))
        features.append(float(charm_exposure / max_exposure))
        
        # Add derived features (ratios between Greeks)
        features.append(float(delta_exposure / (gamma_exposure + 1e-6)))
        features.append(float(vanna_exposure / (charm_exposure + 1e-6)))
        
        # Add momentum features
        features.append(float(momentum_data.get('momentum_score', 0)))
        
        # Map energy direction to numeric value
        energy_map = {
            'UP': 1.0, 'DOWN': -1.0, 'FLAT': 0.0, 
            'ACCELERATING': 0.8, 'DECELERATING': -0.8,
            'BUILDING': 0.5, 'REVERSING': -0.5
        }
        features.append(float(energy_map.get(momentum_data.get('energy_direction', 'FLAT'), 0.0)))
        
        # Add volatility features
        features.append(float(momentum_data.get('volatility', 0.2)))
        features.append(float(momentum_data.get('volatility_trend', 0)))
        
        # Map market regime to numeric value
        regime_map = {
            'BULLISH': 1.0, 'BEARISH': -1.0, 'NEUTRAL': 0.0,
            'VOLATILE': 0.5, 'CONSOLIDATING': -0.5, 'COMPLEX': 0.0
        }
        features.append(float(regime_map.get(greek_results.get('market_regime', 'COMPLEX'), 0.5)))
        
        # Count anomalies
        anomalies = greek_results.get('anomalies', [])
        features.append(float(len(anomalies)))
        
        # Add specific anomaly flags
        anomaly_types = ['GAMMA_SPIKE', 'VANNA_SURGE', 'VANNA_COLLAPSE', 
                         'VOL_COMPRESSION', 'GAMMA_WALL', 'CHARM_ACCELERATION',
                         'DELTA_FLIP', 'GAMMA_REVERSAL']
        
        for atype in anomaly_types:
            features.append(1.0 if any(a.get('type') == atype for a in anomalies) else 0.0)
        
        # Log the number of features extracted
        logger.info(f"Extracted {len(features)} features")
        
        return features
    
    def _predict_with_rules(self, greek_results, momentum_data):
        """
        Predict pattern using rule-based approach.
        
        Args:
            greek_results (dict): Results from Greek energy flow analysis.
            momentum_data (dict): Momentum indicators.
            
        Returns:
            dict: Predicted pattern with confidence and description.
        """
        # Extract key information
        market_regime = greek_results.get('market_regime', 'COMPLEX')
        energy_direction = momentum_data.get('energy_direction', 'FLAT')
        anomalies = [a.get('type') for a in greek_results.get('anomalies', [])]
        
        # Calculate match scores for each pattern
        pattern_scores = {}
        
        for pattern_name, pattern_info in self.patterns.items():
            score = 0
            max_score = 0
            
            # Check market regime match
            if 'conditions' in pattern_info:
                conditions = pattern_info['conditions']
                
                if 'market_regime' in conditions:
                    max_score += 1
                    if market_regime in conditions['market_regime']:
                        score += 1
                
                if 'energy_direction' in conditions:
                    max_score += 1
                    if energy_direction in conditions['energy_direction']:
                        score += 1
                
                if 'anomalies' in conditions:
                    max_score += len(conditions['anomalies'])
                    for anomaly in conditions['anomalies']:
                        if anomaly in anomalies:
                            score += 1
            
            # Calculate confidence
            confidence = score / max_score if max_score > 0 else 0
            pattern_scores[pattern_name] = confidence
        
        # Find best matching pattern
        best_pattern = max(pattern_scores.items(), key=lambda x: x[1])
        pattern_name, confidence = best_pattern
        
        # If confidence is too low, return COMPLEX pattern
        if confidence < 0.5 and pattern_name != "COMPLEX":
            pattern_name = "COMPLEX"
            confidence = 0.6  # Default confidence for COMPLEX pattern
        
        return {
            "pattern": pattern_name,
            "confidence": confidence,
            "description": self.patterns[pattern_name]["description"]
        }
    
    def train_model(self, training_data, model_type='lightgbm', sequence_data=None):
        """
        Train a pattern recognition model.
        
        Args:
            training_data (list): List of dictionaries with 'greek_results', 'momentum_data', and 'pattern' keys.
            model_type (str): Type of model to train ('lightgbm', 'xgboost', 'lstm').
            sequence_data (np.ndarray, optional): Sequence data for LSTM models.
            
        Returns:
            bool: True if training was successful, False otherwise.
        """
        if not HAS_ML_INTEGRATOR or self.ml_integrator is None:
            logger.error("ML integrator not available")
            return False
        
        try:
            logger.info(f"Training pattern recognition model using {model_type}")
            
            # Extract features and labels
            X = []
            y = []
            pattern_names = list(self.patterns.keys())
            
            for item in training_data:
                features = self._extract_features(item['greek_results'], item['momentum_data'])
                X.append(features)
                
                # Convert pattern name to index
                pattern = item['pattern']
                pattern_idx = pattern_names.index(pattern) if pattern in pattern_names else len(pattern_names) - 1
                y.append(pattern_idx)
            
            # Convert to numpy arrays
            X = np.array(X)
            y = np.array(y)
            
            logger.info(f"Training data: X shape={X.shape}, y shape={y.shape}")
            
            # Train model using ML integrator
            model_path = self.ml_integrator.train_model(X, y, model_type=model_type, sequence_data=sequence_data)
            
            if model_path:
                self.model_path = model_path
                logger.info(f"Successfully trained and saved model to {model_path}")
                return True
            else:
                logger.error("Failed to train model")
                return False
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False
    
    def prepare_sequence_data(self, historical_data, lookback=10):
        """
        Prepare sequence data for LSTM models from historical data.
        
        Args:
            historical_data (list): List of dictionaries with 'greek_results' and 'momentum_data' keys.
            lookback (int): Number of time steps to include in each sequence.
            
        Returns:
            tuple: (sequence_data, labels) for LSTM training.
        """
        if len(historical_data) < lookback + 1:
            logger.error(f"Not enough historical data for sequence preparation. Need at least {lookback + 1} samples.")
            return None, None
        
        sequences = []
        labels = []
        pattern_names = list(self.patterns.keys())
        
        for i in range(len(historical_data) - lookback):
            # Extract sequence
            seq = []
            for j in range(lookback):
                item = historical_data[i + j]
                features = self._extract_features(item['greek_results'], item['momentum_data'])
                seq.append(features)
            
            # Get label (pattern of the next item after sequence)
            next_item = historical_data[i + lookback]
            pattern = next_item.get('pattern', 'COMPLEX')
            pattern_idx = pattern_names.index(pattern) if pattern in pattern_names else len(pattern_names) - 1
            
            sequences.append(seq)
            labels.append(pattern_idx)
        
        return np.array(sequences), np.array(labels)
    
    def evaluate_models(self, test_data, sequence_data=None):
        """
        Evaluate all available models on test data.
        
        Args:
            test_data (list): List of dictionaries with 'greek_results', 'momentum_data', and 'pattern' keys.
            sequence_data (np.ndarray, optional): Sequence data for LSTM models.
        
        Returns:
            dict: Evaluation results for each model.
        """
        if not HAS_ML_INTEGRATOR or self.ml_integrator is None:
            logger.error("ML integrator not available")
            return {}
        
        try:
            # Extract features and true labels
            X = []
            y_true = []
            pattern_names = list(self.patterns.keys())
            
            for item in test_data:
                features = self._extract_features(item['greek_results'], item['momentum_data'])
                X.append(features)
                
                # Convert pattern name to index
                pattern = item['pattern']
                pattern_idx = pattern_names.index(pattern) if pattern in pattern_names else len(pattern_names) - 1
                y_true.append(pattern_idx)
            
            # Convert to numpy arrays
            X = np.array(X)
            y_true = np.array(y_true)
            
            # Evaluate each model
            results = {}
            for model_name, model in self.ml_integrator.models.items():
                model_results = {
                    'accuracy': 0,
                    'predictions': [],
                    'confusion_matrix': None
                }
                
                # Make predictions
                y_pred = []
                for i in range(len(X)):
                    features = X[i:i+1]  # Single sample
                    seq = None if sequence_data is None else sequence_data[i:i+1]
                    
                    # Get prediction
                    if model_name.endswith('pt') and seq is not None:
                        # LSTM model
                        pred = self.ml_integrator._predict_with_lstm_model(model_name, model, seq)
                    else:
                        # Standard model
                        pred = self.ml_integrator._predict_with_standard_model(model_name, model, features)
                    
                    if pred is not None:
                        pred_idx, confidence = pred
                        y_pred.append(pred_idx)
                        model_results['predictions'].append({
                            'true': y_true[i],
                            'pred': pred_idx,
                            'confidence': confidence
                        })
                    else:
                        y_pred.append(-1)  # Invalid prediction
                
                # Calculate accuracy
                if y_pred:
                    correct = sum(1 for i in range(len(y_pred)) if y_pred[i] == y_true[i])
                    model_results['accuracy'] = correct / len(y_pred)
                
                # Add to results
                results[model_name] = model_results
            
            return results
            
        except Exception as e:
            logger.error(f"Error evaluating models: {e}")
            return {}
    
    def get_model_diagnostics(self):
        """
        Get diagnostic information about all loaded models.
        
        Returns:
            dict: Diagnostic information for each model.
        """
        if not HAS_ML_INTEGRATOR or self.ml_integrator is None:
            return {"status": "ML integrator not available"}
        
        try:
            # Get diagnostics from ML integrator
            model_diagnostics = self.ml_integrator.diagnose_models()
            
            # Add pattern analyzer specific diagnostics
            for model_name in model_diagnostics:
                # Add last used timestamp if available
                if hasattr(self, '_last_used') and model_name in self._last_used:
                    model_diagnostics[model_name]['last_used'] = self._last_used[model_name]
                
                # Add performance metrics if available
                if hasattr(self, '_model_performance') and model_name in self._model_performance:
                    model_diagnostics[model_name]['performance'] = self._model_performance[model_name]
            
            return model_diagnostics
        
        except Exception as e:
            logger.error(f"Error getting model diagnostics: {e}")
            return {"error": str(e)}
