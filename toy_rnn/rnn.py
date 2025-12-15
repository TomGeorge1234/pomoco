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


class MultiSetRNN(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_heads = nn.ModuleDict()
        self.output_heads = nn.ModuleDict()
        
        self.lstm_backbone = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.shared_proj = nn.Linear(hidden_size * 2, hidden_size)

    def add_dataset(self, dataset_id, input_size, output_size):
        sid = str(dataset_id)
        self.input_heads[sid] = nn.Linear(input_size, self.hidden_size)
        self.output_heads[sid] = nn.Linear(self.hidden_size, output_size)

    def forward(self, x, dataset_id):
        # dataset_id is now a SINGLE value, not a tensor of IDs
        sid = str(dataset_id)
        
        # 1. Select Heads directly
        input_layer = self.input_heads[sid]
        output_layer = self.output_heads[sid]
        
        # 2. Straight-line execution (No masking! No slicing!)
        # x shape: [Batch, Time, Input_Size_Of_This_Dataset]
        emb = input_layer(x)
        
        rnn_out, _ = self.lstm_backbone(emb)
        shared_feats = self.shared_proj(rnn_out)
        
        predictions = output_layer(shared_feats)
        
        return predictions
    

# Example usage
if __name__ == "__main__":
    # Embedding layer handled separately
    embedding = nn.Linear(10, 64)  # raw_input_size=10 -> embedding_size=64
    model = BidirectionalLSTM(input_size=64, hidden_size=64, num_layers=2)
    
    x = torch.randn(4, 20, 10)  # batch_size=4, seq_length=20, raw_input_size=10
    x = embedding(x)  # (4, 20, 64)
    output, (hidden, cell) = model(x)
    print(f"Output shape: {output.shape}")
