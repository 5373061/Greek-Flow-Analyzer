import unittest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn.model_selection import cross_val_score
# Try to import XGBoost
xgboost_missing = True
try:
    from xgboost import XGBRegressor
    xgboost_missing = False
except ImportError:
    pass
from lightgbm import LGBMClassifier
import optuna
import pytest

# Skip this test if pytorch is not installed
pytorch_missing = True
try:
    import torch
    import torch.nn as nn
    pytorch_missing = False
except ImportError:
    pass

@pytest.mark.skipif(pytorch_missing, reason="PyTorch not installed")
class TestOptionsMLModels(unittest.TestCase):
    def setUp(self):
        """Initialize test environment with synthetic options data"""
        self.lookback = 10
        self.forecast_horizon = 3
        
        # Generate synthetic data for testing
        self.options_data = self._generate_options_data()

    def _generate_options_data(self):
        """Generate synthetic options data for testing"""
        # Create a DataFrame with synthetic data
        np.random.seed(42)
        dates = pd.date_range(start='2022-01-01', periods=100)
        
        data = {
            'date': np.repeat(dates, 10),
            'strike': np.tile(np.linspace(90, 110, 10), 100),
            'expiration': np.repeat([d + pd.Timedelta(days=30) for d in dates], 10),
            'type': np.tile(['call', 'put'] * 5, 100),
            'price': np.random.uniform(1, 10, 1000),
            'underlying_price': np.repeat(np.random.normal(100, 5, 100), 10),
            'implied_volatility': np.random.uniform(0.1, 0.5, 1000),
            'delta': np.random.uniform(-1, 1, 1000),
            'gamma': np.random.uniform(0, 0.1, 1000),
            'vega': np.random.uniform(0, 1, 1000),
            'theta': np.random.uniform(-1, 0, 1000),
            'rho': np.random.uniform(-0.5, 0.5, 1000),
            'open_interest': np.random.randint(10, 1000, 1000),
            'volume': np.random.randint(1, 500, 1000)
        }
        
        return pd.DataFrame(data)

    def test_lstm_iv_prediction(self):
        """Test LSTM for implied volatility prediction"""
        if pytorch_missing:
            self.skipTest("PyTorch not installed")
            
        class LSTMModel(nn.Module):
            def __init__(self, input_size, hidden_size=32):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
                self.dropout = nn.Dropout(0.2)
                self.fc = nn.Linear(hidden_size, 1)
                
            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                lstm_out = lstm_out[:, -1, :]  # Take the last output
                x = self.dropout(lstm_out)
                x = self.fc(x)
                return x
        
        features = [
            'delta', 'gamma', 'vega', 'theta',
            'open_interest', 'volume'
        ]
        
        X, y = self._prepare_sequence_data(features, target='implied_volatility')
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        # Create dataset and dataloader
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16)
        
        # Initialize model
        model = LSTMModel(input_size=len(features))
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Train model
        initial_loss = 0
        final_loss = 0
        
        for epoch in range(10):  # Reduced epochs for faster testing
            model.train()
            epoch_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y.unsqueeze(1))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            if epoch == 0:
                initial_loss = epoch_loss / len(train_loader)
            if epoch == 9:
                final_loss = epoch_loss / len(train_loader)
        
        # Verify learning
        self.assertLess(final_loss, initial_loss, "Model should show learning progress")

    @pytest.mark.skipif(xgboost_missing, reason="XGBoost not installed")
    def test_xgboost_price_prediction(self):
        """Test XGBoost for option price prediction"""
        if xgboost_missing:
            self.skipTest("XGBoost not installed")
            
        model = XGBRegressor(
            n_estimators=50,  # Reduced for faster testing
            learning_rate=0.1,
            max_depth=3,
            objective='reg:squarederror'
        )
        
        features = [
            'strike', 'delta', 'gamma', 'vega', 'theta',
            'implied_volatility', 'open_interest'
        ]
        
        # Prepare data
        df = self.options_data.copy()
        X = df[features].values
        y = df['price'].values
        
        # Train-test split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predict and evaluate
        y_pred = model.predict(X_test)
        mse = np.mean((y_pred - y_test) ** 2)
        
        # Feature importance analysis
        importances = model.feature_importances_
        self.assertGreater(
            importances[features.index('implied_volatility')],
            np.mean(importances),
            "Implied volatility should be a significant feature for price prediction"
        )
        
        # Check that model has reasonable accuracy
        self.assertLess(mse, np.var(y_test), "Model should perform better than predicting mean")

    def test_lightgbm_option_type_classification(self):
        """Test LightGBM for classifying profitable option types"""
        # LightGBM for classification
        model = LGBMClassifier(
            n_estimators=50,  # Reduced for faster testing
            learning_rate=0.1,
            max_depth=3
        )
        
        # Create target: 1 if call option is more profitable than put, 0 otherwise
        df = self.options_data.copy()
        df['moneyness'] = df['underlying_price'] / df['strike']
        df['days_to_expiry'] = (df['expiration'] - df['date']).dt.days
        
        # Synthetic profitability based on option characteristics
        df['call_profit'] = (df['moneyness'] - 1) * df['days_to_expiry'] / 30
        df['put_profit'] = (1 - df['moneyness']) * df['days_to_expiry'] / 30
        df['target'] = (df['call_profit'] > df['put_profit']).astype(int)
        
        features = [
            'moneyness', 'days_to_expiry', 'implied_volatility',
            'delta', 'gamma', 'open_interest', 'volume'
        ]
        
        X = df[features].values
        y = df['target'].values
        
        # Train-test split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predict and evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Check that model has reasonable accuracy
        self.assertGreater(accuracy, 0.6, "Model should have accuracy better than random guessing")
        
        # Feature importance analysis
        importances = model.feature_importances_
        self.assertTrue(
            np.max(importances) > np.mean(importances),
            "Model should identify important features"
        )

    def _prepare_sequence_data(self, features, target='implied_volatility'):
        """Prepare sequential data for LSTM"""
        data = self.options_data
        
        # Group by date to create sequences
        grouped = data.groupby('date')
        dates = sorted(data['date'].unique())
        
        sequences = []
        targets = []
        
        for i in range(len(dates) - self.lookback - self.forecast_horizon):
            # Get data for lookback period
            seq_data = []
            for j in range(self.lookback):
                date_data = grouped.get_group(dates[i + j])
                # Average the features for this date
                seq_data.append(date_data[features].mean().values)
            
            # Target is the average implied volatility after forecast_horizon
            target_date = dates[i + self.lookback + self.forecast_horizon]
            target_data = grouped.get_group(target_date)
            target_value = target_data[target].mean()
            
            sequences.append(seq_data)
            targets.append(target_value)
            
        return np.array(sequences), np.array(targets)

if __name__ == '__main__':
    unittest.main()