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
    def __init__(self, num_sets, input_size, hidden_size, latent_dim):
        super().__init__()
        
        # 1. Dataset-Specific Input Encoders
        # We store them in a ModuleList so PyTorch registers the parameters
        # Each maps: [num_neurons] -> [hidden_size] (or an intermediate embedding size)
        self.encoders = nn.ModuleList([
            nn.Linear(input_size, hidden_size) for _ in range(num_sets)
        ])
        
        # 2. Shared Backbone
        self.lstm_backbone = BidirectionalLSTM(
            input_size=hidden_size, # RNN takes the encoded embedding
            hidden_size=hidden_size
        )
        
        # 3. Shared Readout/Decoder
        # Maps RNN output (hidden_size * 2 due to bidirectional) -> latents
        self.readout = nn.Linear(hidden_size * 2, latent_dim)

    def forward(self, x, dataset_indices):
        """
        x: (Batch, Time, Neurons)
        dataset_indices: (Batch) - Integer IDs of which dataset each sample belongs to
        """        
        batch_size, seq_len, _num_neurons = x.size()
        
        # Prepare a placeholder for the embedded input
        # Shape: (Batch, Time, Hidden_Size)
        embedding_output = torch.zeros(
            batch_size, seq_len, self.encoders[0].out_features, 
            device=x.device
        )

        # ---------------------------------------------------------
        # ROUTING LOGIC:
        # Since a batch is mixed, we iterate through the unique set IDs 
        # present in this batch and apply the specific layer to relevant samples.
        # ---------------------------------------------------------
        unique_sets = torch.unique(dataset_indices)
        
        for set_id in unique_sets:
            # Create a boolean mask for items belonging to this set
            mask = (dataset_indices == set_id)
            
            # Select inputs for this set
            # x[mask] shape becomes (Num_Samples_in_Set, Time, Neurons)
            subset_input = x[mask]
            
            # Pass through the specific encoder for this set
            subset_embedded = self.encoders[set_id](subset_input)
            
            # Place result back into the main storage tensor
            embedding_output[mask] = subset_embedded

        # 4. Pass combined embeddings through Shared RNN
        # rnn_out shape: (Batch, Time, Hidden*2)
        rnn_out, _ = self.lstm_backbone(embedding_output)
        
        # 5. Pass through Shared Readout
        predictions = self.readout(rnn_out)
        
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
