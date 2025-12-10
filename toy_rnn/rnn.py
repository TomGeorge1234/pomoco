import torch
import torch.nn as nn

class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(BidirectionalLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
        )
    
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        lstm_out, (hidden, cell) = self.lstm(x)
        # lstm_out shape: (batch_size, seq_length, hidden_size * 2)
        return lstm_out, (hidden, cell)


class MultiInputModel(nn.Module):
    def __init__(
            self, 
            ):
        super().__init__()




# Example usage
if __name__ == "__main__":
    # Embedding layer handled separately
    embedding = nn.Linear(10, 64)  # raw_input_size=10 -> embedding_size=64
    model = BidirectionalLSTM(input_size=64, hidden_size=64, num_layers=2)
    
    x = torch.randn(4, 20, 10)  # batch_size=4, seq_length=20, raw_input_size=10
    x = embedding(x)  # (4, 20, 64)
    output, (hidden, cell) = model(x)
    print(f"Output shape: {output.shape}")
