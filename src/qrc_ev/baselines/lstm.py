"""B2: LSTM baseline for time-series forecasting.

Long Short-Term Memory network using PyTorch.
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class LSTMForecaster:
    """LSTM network for time-series forecasting.
    
    A multi-layer LSTM with configurable architecture,
    trained with Adam optimizer and early stopping.
    
    Attributes:
        hidden_size: Number of LSTM hidden units.
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout rate between layers.
        learning_rate: Adam learning rate.
        epochs: Maximum training epochs.
        patience: Early stopping patience.
    """
    
    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        epochs: int = 100,
        patience: int = 10,
        batch_size: int = 32,
        seq_length: int = 24,
        seed: int = 42,
    ):
        """Initialize the LSTM forecaster.
        
        Args:
            hidden_size: LSTM hidden units. Default: 64.
            num_layers: Number of LSTM layers. Default: 2.
            dropout: Dropout rate. Default: 0.2.
            learning_rate: Learning rate. Default: 0.001.
            epochs: Max epochs. Default: 100.
            patience: Early stopping patience. Default: 10.
            batch_size: Training batch size. Default: 32.
            seq_length: Input sequence length. Default: 24.
            seed: Random seed. Default: 42.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for LSTM. Install with: pip install torch")
            
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.seed = seed
        
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._input_dim = None
        
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    
    def _build_model(self, input_dim: int):
        """Build the LSTM model."""
        
        class LSTMModel(nn.Module):
            def __init__(self, input_dim, hidden_size, num_layers, dropout):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=input_dim,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout if num_layers > 1 else 0,
                    batch_first=True,
                )
                self.fc = nn.Linear(hidden_size, 1)
                
            def forward(self, x):
                # x: (batch, seq_len, input_dim)
                lstm_out, _ = self.lstm(x)
                # Take last timestep
                out = self.fc(lstm_out[:, -1, :])
                return out.squeeze(-1)
        
        self.model = LSTMModel(input_dim, self.hidden_size, self.num_layers, self.dropout)
        self.model.to(self.device)
        self._input_dim = input_dim
        
    def _create_sequences(self, X: np.ndarray, y: np.ndarray):
        """Create sequences for LSTM training."""
        sequences = []
        targets = []
        
        for i in range(len(X) - self.seq_length):
            sequences.append(X[i:i + self.seq_length])
            targets.append(y[i + self.seq_length])
            
        return np.array(sequences), np.array(targets)
    
    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None):
        """Fit the LSTM to training data.
        
        Args:
            X: Input features of shape (T, input_dim).
            y: Target values of shape (T,).
            X_val: Optional validation features.
            y_val: Optional validation targets.
        """
        if self.model is None:
            self._build_model(X.shape[1])
            
        # Create sequences
        X_seq, y_seq = self._create_sequences(X, y)
        
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        y_tensor = torch.FloatTensor(y_seq).to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Validation data
        if X_val is not None and y_val is not None:
            X_val_seq, y_val_seq = self._create_sequences(X_val, y_val)
            X_val_tensor = torch.FloatTensor(X_val_seq).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val_seq).to(self.device)
        else:
            X_val_tensor = y_val_tensor = None
            
        # Training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0
            
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                pred = self.model(batch_X)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
            train_loss /= len(loader)
            
            # Validation
            if X_val_tensor is not None:
                self.model.eval()
                with torch.no_grad():
                    val_pred = self.model(X_val_tensor)
                    val_loss = criterion(val_pred, y_val_tensor).item()
                    
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
                    
            if (epoch + 1) % 10 == 0:
                val_str = f", Val Loss: {val_loss:.4f}" if X_val_tensor is not None else ""
                print(f"Epoch {epoch + 1}/{self.epochs}, Train Loss: {train_loss:.4f}{val_str}")
                
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the trained LSTM.
        
        Args:
            X: Input features of shape (T, input_dim).
            
        Returns:
            Predictions of shape (T - seq_length,).
        """
        self.model.eval()
        
        # Create sequences
        sequences = []
        for i in range(len(X) - self.seq_length):
            sequences.append(X[i:i + self.seq_length])
        sequences = np.array(sequences)
        
        X_tensor = torch.FloatTensor(sequences).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()
            
        return predictions
