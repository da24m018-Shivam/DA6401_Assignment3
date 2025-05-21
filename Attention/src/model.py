import tensorflow as tf
import numpy as np

class AttentionMechanism(tf.keras.layers.Layer):
    def __init__(self, units, method='luong'):
        super(AttentionMechanism, self).__init__()
        self.units = units
        self.method = method.lower()
        
        if self.method == 'bahdanau':
            # Bahdanau attention (additive)
            self.W1 = tf.keras.layers.Dense(units)
            self.W2 = tf.keras.layers.Dense(units)
            self.V = tf.keras.layers.Dense(1)
        # No parameters needed for Luong attention (multiplicative)
    
    def call(self, query, values):
        # query shape: [batch_size, hidden_size]
        # values shape: [batch_size, seq_len, hidden_size]
        
        if self.method == 'bahdanau':
            # Bahdanau attention
            # Expand query to [batch_size, 1, hidden_size]
            query_expanded = tf.expand_dims(query, 1)
            
            # Calculate score
            score = self.V(tf.nn.tanh(self.W1(query_expanded) + self.W2(values)))
            
            # Remove the last dimension
            score = tf.squeeze(score, axis=-1)
        else:
            # Luong attention (multiplicative)
            # Reshape query to [batch_size, hidden_size, 1]
            query_reshaped = tf.expand_dims(query, -1)
            
            # Calculate score: [batch_size, seq_len, 1]
            score = tf.matmul(values, query_reshaped)
            
            # Remove the last dimension
            score = tf.squeeze(score, axis=-1)
        
        # Apply softmax to get attention weights
        attention_weights = tf.nn.softmax(score, axis=1)
        
        # Calculate context vector
        context = tf.matmul(tf.expand_dims(attention_weights, 1), values)
        context = tf.squeeze(context, axis=1)
        
        return context, attention_weights

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, hidden_units, cell_type='gru', num_layers=1, dropout_rate=0.0):
        super(Encoder, self).__init__()
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.cell_type = cell_type.lower()
        
        # Embedding layer
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        
        # RNN layers
        self.rnn_cells = []
        for i in range(num_layers):
            if self.cell_type == 'lstm':
                cell = tf.keras.layers.LSTM(
                    hidden_units, 
                    return_sequences=True, 
                    return_state=True, 
                    dropout=dropout_rate if i < num_layers-1 else 0.0,
                    recurrent_initializer='glorot_uniform'
                )
            elif self.cell_type == 'gru':
                cell = tf.keras.layers.GRU(
                    hidden_units, 
                    return_sequences=True, 
                    return_state=True, 
                    dropout=dropout_rate if i < num_layers-1 else 0.0,
                    recurrent_initializer='glorot_uniform'
                )
            else:  # Simple RNN
                cell = tf.keras.layers.SimpleRNN(
                    hidden_units, 
                    return_sequences=True, 
                    return_state=True, 
                    dropout=dropout_rate if i < num_layers-1 else 0.0,
                    recurrent_initializer='glorot_uniform'
                )
            self.rnn_cells.append(cell)
    
    def call(self, x, training=False):
        # x shape: [batch_size, seq_len]
        
        # Embedding
        x = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        
        # Process through RNN layers
        states = []
        for i, rnn_cell in enumerate(self.rnn_cells):
            if self.cell_type == 'lstm':
                x, state_h, state_c = rnn_cell(x, training=training)
                states.append([state_h, state_c])
            else:
                x, state = rnn_cell(x, training=training)
                states.append(state)
        
        return x, states

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, hidden_units, cell_type='gru', 
                 num_layers=1, dropout_rate=0.0, attention_type='luong'):
        super(Decoder, self).__init__()
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.cell_type = cell_type.lower()
        self.vocab_size = vocab_size
        
        # Embedding layer
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        
        # Attention mechanism
        self.attention = AttentionMechanism(hidden_units, attention_type)
        
        # RNN layers
        self.rnn_cells = []
        for i in range(num_layers):
            if self.cell_type == 'lstm':
                cell = tf.keras.layers.LSTM(
                    hidden_units, 
                    return_sequences=True, 
                    return_state=True, 
                    dropout=dropout_rate if i < num_layers-1 else 0.0,
                    recurrent_initializer='glorot_uniform'
                )
            elif self.cell_type == 'gru':
                cell = tf.keras.layers.GRU(
                    hidden_units, 
                    return_sequences=True, 
                    return_state=True, 
                    dropout=dropout_rate if i < num_layers-1 else 0.0,
                    recurrent_initializer='glorot_uniform'
                )
            else:  # Simple RNN
                cell = tf.keras.layers.SimpleRNN(
                    hidden_units, 
                    return_sequences=True, 
                    return_state=True, 
                    dropout=dropout_rate if i < num_layers-1 else 0.0,
                    recurrent_initializer='glorot_uniform'
                )
            self.rnn_cells.append(cell)
        
        # Output layer
        self.fc = tf.keras.layers.Dense(vocab_size)
        
        # Dropout
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
    
    def call(self, x, hidden, encoder_output, training=False):
        # x shape: [batch_size]
        # hidden shape: list of states for each layer
        # encoder_output shape: [batch_size, seq_len, hidden_units]
        
        # Get the last hidden state
        if self.cell_type == 'lstm':
            query = hidden[-1][0]  # Use hidden state, not cell state
        else:
            query = hidden[-1]
        
        # Calculate attention
        context, attention_weights = self.attention(query, encoder_output)
        
        # Expand x to [batch_size, 1]
        x = tf.expand_dims(x, 1)
        
        # Embedding
        x = self.embedding(x)  # [batch_size, 1, embedding_dim]
        
        # Concatenate context vector and embedding
        x = tf.concat([tf.expand_dims(context, 1), x], axis=-1)
        
        # Apply dropout
        x = self.dropout(x, training=training)
        
        # Process through RNN layers
        new_hidden = []
        for i, rnn_cell in enumerate(self.rnn_cells):
            if i > 0:
                x = self.dropout(x, training=training)
            
            if self.cell_type == 'lstm':
                current_hidden = hidden[i]
                x, state_h, state_c = rnn_cell(x, initial_state=current_hidden, training=training)
                new_hidden.append([state_h, state_c])
            else:
                current_hidden = hidden[i]
                x, state = rnn_cell(x, initial_state=current_hidden, training=training)
                new_hidden.append(state)
        
        # Output
        x = self.fc(tf.concat([x[:, 0], context], axis=-1))
        
        return x, new_hidden, attention_weights

class Seq2SeqModel(tf.keras.Model):
    def __init__(self, encoder, decoder, start_token_idx, end_token_idx):
        super(Seq2SeqModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.start_token_idx = start_token_idx
        self.end_token_idx = end_token_idx
    
    def build(self, input_shape):
        # This method explicitly builds the model
        # input_shape is a tuple of (source_shape, target_shape)
        source_shape, target_shape = input_shape
        
        # Build encoder
        self.encoder.build(source_shape)
        
        # Build decoder - create a dummy state to help build it
        dummy_source = tf.zeros((1,) + source_shape[1:], dtype=tf.int32)
        dummy_encoder_output, dummy_states = self.encoder(dummy_source)
        
        # Build decoder with dummy inputs
        dummy_decoder_input = tf.zeros((1,), dtype=tf.int32)
        self.decoder.build([(1,), dummy_states[0].shape, dummy_encoder_output.shape])
        
        self.built = True
    
    def call(self, inputs, training=False):
        # Unpack inputs
        source, target = inputs
        
        batch_size = tf.shape(source)[0]
        target_length = tf.shape(target)[1]
        
        # Encode the source sequence
        encoder_output, encoder_states = self.encoder(source, training=training)
        
        # Initialize output tensor - make sure it has the right shape
        outputs = tf.TensorArray(tf.float32, size=target_length-1)
        
        # Prepare decoder input and states
        decoder_input = tf.fill([batch_size], self.start_token_idx)
        decoder_states = encoder_states
        
        # Teacher forcing ratio
        teacher_forcing_ratio = 0.5 if training else 0.0
        
        # Decode step by step - start from position 1 (after start token)
        for t in range(1, target_length):
            # Get output from decoder
            prediction, decoder_states, _ = self.decoder(
                decoder_input, decoder_states, encoder_output, training=training)
            
            # Store prediction
            outputs = outputs.write(t-1, prediction)
            
            # Teacher forcing
            if training and tf.random.uniform([]) < teacher_forcing_ratio:
                decoder_input = target[:, t]
            else:
                decoder_input = tf.argmax(prediction, axis=1)
        
        # Stack outputs - ensure it has the right shape to match target
        outputs = tf.transpose(outputs.stack(), [1, 0, 2])
        
        return outputs
    def translate(self, source, max_length=50):
        """
        Translate source sequences to target sequences
        
        Args:
            source: Source sequence tensor [batch_size, seq_len]
            max_length: Maximum length of generated sequence
            
        Returns:
            translations: Generated sequences [batch_size, seq_len]
            attention_weights: Attention weights for visualization [batch_size, tgt_len, src_len]
        """
        batch_size = tf.shape(source)[0]
        
        # Encode the source sequence
        encoder_output, encoder_states = self.encoder(source, training=False)
        
        # Prepare decoder input and states
        decoder_input = tf.fill([batch_size], self.start_token_idx)
        decoder_states = encoder_states
        
        # Initialize result tensors - specify int32 dtype
        translations = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
        attention_weights_array = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        
        # Store first token (start token)
        translations = translations.write(0, decoder_input)
        
        for t in range(1, max_length):
            # Get output from decoder
            prediction, decoder_states, attention_weights = self.decoder(
                decoder_input, decoder_states, encoder_output, training=False)
            
            # Get predicted token and cast to int32
            predicted_id = tf.argmax(prediction, axis=1)
            predicted_id = tf.cast(predicted_id, tf.int32)  # Cast to int32
            
            # Store results
            translations = translations.write(t, predicted_id)
            attention_weights_array = attention_weights_array.write(t-1, attention_weights)
            
            # Break if end token is predicted for all sequences in batch
            if tf.reduce_all(tf.equal(predicted_id, self.end_token_idx)):
                break
            
            # Use predicted token as next input
            decoder_input = predicted_id
        
        # Stack results
        translations = tf.transpose(translations.stack())  # [batch_size, seq_len]
        attention_weights = tf.transpose(attention_weights_array.stack(), [1, 0, 2])  # [batch_size, tgt_len, src_len]
        
        return translations, attention_weights




def create_model(source_vocab_size, target_vocab_size, embedding_dim=256, hidden_units=512, 
                cell_type='gru', num_layers=2, dropout_rate=0.1, attention_type='luong',
                start_token_idx=0, end_token_idx=1):
    encoder = Encoder(
        source_vocab_size, embedding_dim, hidden_units, 
        cell_type, num_layers, dropout_rate
    )
    
    decoder = Decoder(
        target_vocab_size, embedding_dim, hidden_units, 
        cell_type, num_layers, dropout_rate, attention_type
    )
    
    model = Seq2SeqModel(encoder, decoder, start_token_idx, end_token_idx)
    
    return model
