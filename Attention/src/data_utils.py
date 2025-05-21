import tensorflow as tf
import os
import numpy as np

class TransliterationDataset(tf.keras.utils.Sequence):
    def __init__(self, source_tensor, target_tensor, batch_size=64, shuffle=True):
        self.source_tensor = source_tensor
        self.target_tensor = target_tensor
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.source_tensor))
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return len(self.source_tensor) // self.batch_size
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        return self.source_tensor[batch_indices], self.target_tensor[batch_indices]
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

def load_corpus(filepath):
    """Extract transliteration pairs from corpus file"""
    corpus_entries = []
    with open(filepath, 'r', encoding='utf-8') as corpus_file:
        for entry in corpus_file:
            segments = entry.strip().split('\t')
            if len(segments) >= 2:
                roman_text = segments[1]
                native_text = segments[0]
                corpus_entries.append((roman_text, native_text))
    return corpus_entries

def create_vocabulary(corpus_entries):
    """Generate character-to-index mappings from corpus"""
    roman_character_set = set()
    native_character_set = set()
    
    for roman_text, native_text in corpus_entries:
        roman_character_set.update(roman_text)
        native_character_set.update(native_text)
    
    print("Sample native script chars:", sorted(native_character_set)[:10])
    print("Sample roman script chars:", sorted(roman_character_set)[:10])
    
    # Add special tokens
    special_tokens = ['<start>', '<end>', '<pad>']
    for token in special_tokens:
        roman_character_set.add(token)
        native_character_set.add(token)
    
    # Create dictionaries
    roman_to_idx = {char: idx for idx, char in enumerate(sorted(roman_character_set))}
    idx_to_roman = {idx: char for char, idx in roman_to_idx.items()}
    
    native_to_idx = {char: idx for idx, char in enumerate(sorted(native_character_set))}
    idx_to_native = {idx: char for char, idx in native_to_idx.items()}
    
    return roman_to_idx, idx_to_roman, native_to_idx, idx_to_native

def encode_sequences(corpus_entries, roman_to_idx, native_to_idx, max_roman_len=None, max_native_len=None):
    """Convert character sequences to index sequences"""
    if max_roman_len is None:
        max_roman_len = max(len(roman) for roman, _ in corpus_entries) + 2  # +2 for <start> and <end>
    
    if max_native_len is None:
        max_native_len = max(len(native) for _, native in corpus_entries) + 2
    
    roman_sequences = []
    native_sequences = []
    
    for roman_text, native_text in corpus_entries:
        # Process roman text
        roman_indices = [roman_to_idx['<start>']]
        roman_indices.extend(roman_to_idx[char] for char in roman_text)
        roman_indices.append(roman_to_idx['<end>'])
        
        # Pad roman sequence
        padding_length = max_roman_len - len(roman_indices)
        roman_indices.extend([roman_to_idx['<pad>']] * padding_length)
        roman_sequences.append(roman_indices)
        
        # Process native text
        native_indices = [native_to_idx['<start>']]
        native_indices.extend(native_to_idx[char] for char in native_text)
        native_indices.append(native_to_idx['<end>'])
        
        # Pad native sequence
        padding_length = max_native_len - len(native_indices)
        native_indices.extend([native_to_idx['<pad>']] * padding_length)
        native_sequences.append(native_indices)
    
    return np.array(roman_sequences), np.array(native_sequences), max_roman_len, max_native_len

def prepare_data_pipeline(corpus_dir, batch_size=64):
    """Setup complete data pipeline for training and evaluation"""
    train_corpus = load_corpus(os.path.join(corpus_dir, 'hi.translit.sampled.train.tsv'))
    dev_corpus = load_corpus(os.path.join(corpus_dir, 'hi.translit.sampled.dev.tsv'))
    test_corpus = load_corpus(os.path.join(corpus_dir, 'hi.translit.sampled.test.tsv'))
    
    # Build character mappings from training and validation data
    roman_to_idx, idx_to_roman, native_to_idx, idx_to_native = create_vocabulary(train_corpus + dev_corpus)
    
    # Transform data to indices
    X_train, y_train, max_roman_len, max_native_len = encode_sequences(
        train_corpus, roman_to_idx, native_to_idx)
    
    X_dev, y_dev, _, _ = encode_sequences(
        dev_corpus, roman_to_idx, native_to_idx, max_roman_len, max_native_len)
    
    X_test, y_test, _, _ = encode_sequences(
        test_corpus, roman_to_idx, native_to_idx, max_roman_len, max_native_len)
    
    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train)).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_dev, y_dev)).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)
    
    return {
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset,
        'roman_to_idx': roman_to_idx,
        'idx_to_roman': idx_to_roman,
        'native_to_idx': native_to_idx,
        'idx_to_native': idx_to_native,
        'max_roman_len': max_roman_len,
        'max_native_len': max_native_len
    }
