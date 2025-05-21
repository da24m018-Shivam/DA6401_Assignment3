"""Configuration parameters for the transliteration model"""

# Default configuration
DEFAULT_CONFIG = {
    'cell_type': 'GRU',
    'embed_size': 128,
    'hidden_size': 128,
    'encoder_layers': 1,
    'decoder_layers': 1,
    'dropout': 0.1,
    'batch_size': 64,
    'learning_rate': 0.0005,
    'teacher_forcing_ratio': 0.3,
    'attention_type': 'bahdanau',  # TensorFlow uses 'luong' instead of 'dot'
    'clip_norm': 1.0,
    'epochs': 1
}

def calculate_model_complexity(embedding_size, hidden_size, sequence_length, vocabulary_size):
    """
    Calculate model complexity
    
    Parameters:
    embedding_size: Size of embedding vectors
    hidden_size: Size of hidden state
    sequence_length: Maximum sequence length
    vocabulary_size: Size of vocabulary
    """
    # Embedding layer parameters
    embedding_params = 2 * vocabulary_size * embedding_size  # Source and target embeddings
    
    # RNN parameters (for both encoder and decoder)
    encoder_rnn_params = embedding_size * hidden_size + hidden_size * hidden_size + hidden_size  # Input-to-hidden, hidden-to-hidden, bias
    decoder_rnn_params = embedding_size * hidden_size + hidden_size * hidden_size + hidden_size
    
    # Output layer parameters
    output_params = hidden_size * vocabulary_size + vocabulary_size  # Hidden-to-output, bias
    
    # Total parameters
    total_params = embedding_params + encoder_rnn_params + decoder_rnn_params + output_params
    
    # Computations
    # Encoder computations
    encoder_comp = sequence_length * (embedding_size * hidden_size + hidden_size * hidden_size)
    
    # Decoder computations
    decoder_comp = sequence_length * (embedding_size * hidden_size + hidden_size * hidden_size + hidden_size * vocabulary_size)
    
    # Total computations
    total_comp = encoder_comp + decoder_comp
    
    return {
        'embedding_params': embedding_params,
        'encoder_rnn_params': encoder_rnn_params,
        'decoder_rnn_params': decoder_rnn_params,
        'output_params': output_params,
        'total_params': total_params,
        'encoder_comp': encoder_comp,
        'decoder_comp': decoder_comp,
        'total_comp': total_comp
    }
