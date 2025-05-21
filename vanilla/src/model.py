import torch
import torch.nn as nn
import numpy as np

class EncoderRNN(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, cell_type='RNN', num_layers=1, dropout=0.0):
        """
        Initialize the encoder for sequence-to-sequence model
        
        Args:
            input_size: Size of the input vocabulary
            embed_size: Dimension of the embedding vectors
            hidden_size: Number of hidden units in the RNN
            cell_type: Type of RNN cell ('RNN', 'LSTM', 'GRU')
            num_layers: Number of stacked RNN layers
            dropout: Dropout probability (applied between layers if num_layers > 1)
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, embed_size)
        self.cell_type = cell_type.upper()
        
        # Select appropriate RNN cell based on cell_type
        if self.cell_type == 'LSTM':
            self.rnn = nn.LSTM(
                embed_size, hidden_size, num_layers, 
                batch_first=True, 
                dropout=dropout if num_layers > 1 else 0.0
            )
        elif self.cell_type == 'GRU':
            self.rnn = nn.GRU(
                embed_size, hidden_size, num_layers, 
                batch_first=True, 
                dropout=dropout if num_layers > 1 else 0.0
            )
        else:  # Default to RNN
            self.rnn = nn.RNN(
                embed_size, hidden_size, num_layers, 
                batch_first=True, 
                dropout=dropout if num_layers > 1 else 0.0
            )
        
        # For visualization purposes
        self.activations = None
        self.register_forward_hook(self._get_activations)
    
    def _get_activations(self, module, input, output):
        """Hook to capture activations during forward pass"""
        if self.cell_type == 'LSTM':
            self.activations = output[0].detach()
        else:
            self.activations = output[0].detach()
    
    def forward(self, input_seq):
        """
        Process input sequence through the encoder
        
        Args:
            input_seq: Tensor containing token indices [batch_size, seq_len]
            
        Returns:
            outputs: All hidden states for each input token
            hidden: Final hidden state(s) of the encoder
        """
        # Embed input sequence
        embedded = self.embedding(input_seq)
        
        # Process through RNN
        if self.cell_type == 'LSTM':
            outputs, (hidden, cell) = self.rnn(embedded)
            return outputs, (hidden, cell)
        else:
            outputs, hidden = self.rnn(embedded)
            return outputs, hidden


class DecoderRNN(nn.Module):
    def __init__(self, output_size, embed_size, hidden_size, cell_type='RNN', num_layers=1, dropout=0.0):
        """
        Initialize the decoder for sequence-to-sequence model
        
        Args:
            output_size: Size of the output vocabulary
            embed_size: Dimension of the embedding vectors
            hidden_size: Number of hidden units in the RNN
            cell_type: Type of RNN cell ('RNN', 'LSTM', 'GRU')
            num_layers: Number of stacked RNN layers
            dropout: Dropout probability
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(output_size, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.cell_type = cell_type.upper()
        
        # Select appropriate RNN cell based on cell_type
        if self.cell_type == 'LSTM':
            self.rnn = nn.LSTM(
                embed_size, hidden_size, num_layers, 
                batch_first=True, 
                dropout=dropout if num_layers > 1 else 0.0
            )
        elif self.cell_type == 'GRU':
            self.rnn = nn.GRU(
                embed_size, hidden_size, num_layers, 
                batch_first=True, 
                dropout=dropout if num_layers > 1 else 0.0
            )
        else:  # Default to RNN
            self.rnn = nn.RNN(
                embed_size, hidden_size, num_layers, 
                batch_first=True, 
                dropout=dropout if num_layers > 1 else 0.0
            )
            
        # Output projection layer
        self.out = nn.Linear(hidden_size, output_size)
        
        # For visualization purposes
        self.activations = None
        self.gradients = None
        self.register_forward_hook(self._get_activations)
    
    def _get_activations(self, module, input, output):
        """Hook to capture activations during forward pass"""
        if self.cell_type == 'LSTM':
            self.activations = output[0].detach()
        else:
            self.activations = output[0].detach()
    
    def forward(self, input_char, hidden):
        """
        Process single input token through the decoder
        
        Args:
            input_char: Tensor containing token indices [batch_size]
            hidden: Hidden state from encoder or previous decoder step
            
        Returns:
            output: Prediction scores for each token in vocabulary
            hidden: Updated hidden state(s)
        """
        # Expand dimensions for sequence length of 1
        embedded = self.embedding(input_char).unsqueeze(1)
        embedded = self.dropout(embedded)
        
        # Process through RNN
        if self.cell_type == 'LSTM':
            output, (hidden, cell) = self.rnn(embedded, hidden)
            output = output.squeeze(1)  # Remove sequence dimension
            output = self.out(output)   # Project to vocabulary size
            return output, (hidden, cell)
        else:
            output, hidden = self.rnn(embedded, hidden)
            output = output.squeeze(1)  # Remove sequence dimension
            output = self.out(output)   # Project to vocabulary size
            return output, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        """
        Initialize sequence-to-sequence model
        
        Args:
            encoder: EncoderRNN instance
            decoder: DecoderRNN instance
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
        # For visualization
        self.connectivity_matrix = None
    
    def forward(self, source, target, teacher_forcing_ratio=0.5):
        """
        Run the sequence-to-sequence model
        
        Args:
            source: Source sequence tensor [batch_size, source_len]
            target: Target sequence tensor [batch_size, target_len]
            teacher_forcing_ratio: Probability of using teacher forcing
            
        Returns:
            outputs: Prediction scores for each position [batch_size, target_len, vocab_size]
        """
        batch_size = source.size(0)
        target_len = target.size(1)
        target_vocab_size = self.decoder.out.out_features
        
        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, target_len, target_vocab_size).to(source.device)
        
        # Encode the source sequence
        encoder_outputs, hidden = self.encoder(source)
        
        # First input to the decoder is the start token
        decoder_input = target[:, 0]
        
        # Handle differing encoder/decoder layers
        if self.encoder.cell_type == 'LSTM':
            hidden, cell = hidden
            # If number of layers differ, adjust hidden state dimensions
            if self.encoder.num_layers != self.decoder.num_layers:
                factor = self.decoder.num_layers // self.encoder.num_layers + 1
                hidden = hidden.repeat(factor, 1, 1)[:self.decoder.num_layers]
                cell = cell.repeat(factor, 1, 1)[:self.decoder.num_layers]
            hidden = (hidden, cell)
        else:
            # If number of layers differ, adjust hidden state dimensions
            if self.encoder.num_layers != self.decoder.num_layers:
                factor = self.decoder.num_layers // self.encoder.num_layers + 1
                hidden = hidden.repeat(factor, 1, 1)[:self.decoder.num_layers]
        
        # Generate target sequence one token at a time
        for t in range(1, target_len):
            # Generate prediction for current position
            output, hidden = self.decoder(decoder_input, hidden)
            outputs[:, t, :] = output
            
            # Determine next input: teacher forcing or prediction
            use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
            top_predicted = output.argmax(1)
            decoder_input = target[:, t] if use_teacher_forcing else top_predicted
        
        return outputs
    
    def compute_connectivity(self, source, target):
        """
        Compute connectivity matrix showing which input tokens influence which output tokens
        
        Args:
            source: Source sequence tensor [batch_size, source_len]
            target: Target sequence tensor [batch_size, target_len]
            
        Returns:
            connectivity: Matrix of influence scores [batch_size, target_len, source_len]
        """
        batch_size = source.size(0)
        source_len = source.size(1)
        target_len = target.size(1)
        
        # Initialize connectivity matrix
        connectivity = torch.zeros(batch_size, target_len, source_len)
        
        # Encode the source sequence
        encoder_outputs, hidden = self.encoder(source)
        
        # First input to the decoder is the start token
        decoder_input = target[:, 0]
        
        # Handle differing encoder/decoder layers
        if self.encoder.cell_type == 'LSTM':
            hidden, cell = hidden
            if self.encoder.num_layers != self.decoder.num_layers:
                factor = self.decoder.num_layers // self.encoder.num_layers + 1
                hidden = hidden.repeat(factor, 1, 1)[:self.decoder.num_layers]
                cell = cell.repeat(factor, 1, 1)[:self.decoder.num_layers]
            hidden = (hidden, cell)
        else:
            if self.encoder.num_layers != self.decoder.num_layers:
                factor = self.decoder.num_layers // self.encoder.num_layers + 1
                hidden = hidden.repeat(factor, 1, 1)[:self.decoder.num_layers]
        
        # For each output token, compute gradients with respect to input tokens
        for t in range(1, target_len):
            # Forward pass for this time step
            output, hidden = self.decoder(decoder_input, hidden)
            
            # Get the predicted token
            predicted = output.argmax(1)
            
            # For each example in the batch
            for b in range(batch_size):
                # Create a one-hot vector for the output token
                one_hot = torch.zeros_like(output)
                one_hot[b, predicted[b]] = 1
                
                # Backward pass to get gradients
                self.zero_grad()
                output.backward(one_hot, retain_graph=True)
                
                # Get gradients with respect to encoder outputs
                if hasattr(encoder_outputs, 'grad') and encoder_outputs.grad is not None:
                    # Sum gradients across hidden dimensions
                    grad_magnitudes = encoder_outputs.grad[b].norm(dim=1)
                    connectivity[b, t, :] = grad_magnitudes / grad_magnitudes.sum()
            
            # Use predicted token as next input
            decoder_input = predicted
        
        self.connectivity_matrix = connectivity
        return connectivity