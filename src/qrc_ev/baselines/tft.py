"""B3: Temporal Fusion Transformer (TFT) baseline.

Simplified TFT implementation for time-series forecasting.
Based on the architecture from "Temporal Fusion Transformers 
for Interpretable Multi-horizon Time Series Forecasting" (Lim et al., 2021).
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TemporalFusionTransformer:
    """Simplified Temporal Fusion Transformer for forecasting.
    
    Implements key TFT components:
    - Variable selection networks
    - Gated residual networks
    - Multi-head attention
    - Gated skip connections
    
    Attributes:
        hidden_size: Hidden dimension size.
        num_heads: Number of attention heads.
        dropout: Dropout rate.
        num_layers: Number of transformer layers.
    """
    
    def __init__(
        self,
        hidden_size: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
        num_layers: int = 2,
        learning_rate: float = 0.001,
        epochs: int = 100,
        patience: int = 10,
        batch_size: int = 32,
        seq_length: int = 24,
        seed: int = 42,
    ):
        """Initialize the TFT.
        
        Args:
            hidden_size: Hidden dimension. Default: 64.
            num_heads: Attention heads. Default: 4.
            dropout: Dropout rate. Default: 0.1.
            num_layers: Transformer layers. Default: 2.
            learning_rate: Learning rate. Default: 0.001.
            epochs: Max epochs. Default: 100.
            patience: Early stopping patience. Default: 10.
            batch_size: Batch size. Default: 32.
            seq_length: Input sequence length. Default: 24.
            seed: Random seed. Default: 42.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for TFT. Install with: pip install torch")
            
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.seed = seed
        
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            
    def _build_model(self, input_dim: int):
        """Build the TFT model."""
        
        class GatedResidualNetwork(nn.Module):
            """Gated Residual Network component."""
            def __init__(self, input_dim, hidden_dim, output_dim, dropout):
                super().__init__()
                self.fc1 = nn.Linear(input_dim, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, output_dim)
                self.gate = nn.Linear(hidden_dim, output_dim)
                self.dropout = nn.Dropout(dropout)
                self.layer_norm = nn.LayerNorm(output_dim)
                
                # Skip connection
                if input_dim != output_dim:
                    self.skip = nn.Linear(input_dim, output_dim)
                else:
                    self.skip = None
                    
            def forward(self, x):
                residual = x if self.skip is None else self.skip(x)
                
                h = torch.relu(self.fc1(x))
                h = self.dropout(h)
                
                out = self.fc2(h)
                gate = torch.sigmoid(self.gate(h))
                
                out = gate * out + (1 - gate) * residual
                out = self.layer_norm(out)
                
                return out
        
        class SimpleTFT(nn.Module):
            """Simplified TFT architecture."""
            def __init__(self, input_dim, hidden_size, num_heads, num_layers, dropout):
                super().__init__()
                
                # Input embedding
                self.input_embed = nn.Linear(input_dim, hidden_size)
                
                # Variable selection (simplified)
                self.var_select = GatedResidualNetwork(
                    hidden_size, hidden_size, hidden_size, dropout
                )
                
                # Positional encoding
                self.pos_encoder = nn.Parameter(torch.randn(1, 1000, hidden_size) * 0.01)
                
                # Transformer encoder
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=num_heads,
                    dim_feedforward=hidden_size * 4,
                    dropout=dropout,
                    batch_first=True,
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                
                # Output layers
                self.grn_out = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout)
                self.fc_out = nn.Linear(hidden_size, 1)
                
            def forward(self, x):
                # x: (batch, seq_len, input_dim)
                batch_size, seq_len, _ = x.shape
                
                # Embed input
                h = self.input_embed(x)
                
                # Variable selection
                h = self.var_select(h)
                
                # Add positional encoding
                h = h + self.pos_encoder[:, :seq_len, :]
                
                # Transformer
                h = self.transformer(h)
                
                # Take last timestep
                h = h[:, -1, :]
                
                # Output
                h = self.grn_out(h)
                out = self.fc_out(h)
                
                return out.squeeze(-1)
        
        self.model = SimpleTFT(
            input_dim, self.hidden_size, self.num_heads, 
            self.num_layers, self.dropout
        )
        self.model.to(self.device)
        
    def _create_sequences(self, X: np.ndarray, y: np.ndarray):
        """Create sequences for training."""
        sequences = []
        targets = []
        
        for i in range(len(X) - self.seq_length):
            sequences.append(X[i:i + self.seq_length])
            targets.append(y[i + self.seq_length])
            
        return np.array(sequences), np.array(targets)
    
    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None):
        """Fit the TFT to training data.
        
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
        """Predict using the trained TFT.
        
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
