import torch
from torch.utils.data import Dataset, DataLoader

def load_data(file_path):
    """Load data from the file path"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                latin = parts[1]
                devanagari = parts[0]
                data.append((latin, devanagari))
    return data

def create_vocab(data):
    """Create vocabulary from data"""
    latin_chars = set()
    devanagari_chars = set()
    
    for latin, devanagari in data:
        latin_chars.update(latin)
        devanagari_chars.update(devanagari)
    
    # Add special tokens
    latin_chars.add('<SOS>')
    latin_chars.add('<EOS>')
    latin_chars.add('<PAD>')
    
    devanagari_chars.add('<SOS>')
    devanagari_chars.add('<EOS>')
    devanagari_chars.add('<PAD>')
    
    # Create dictionaries
    latin_to_idx = {char: idx for idx, char in enumerate(sorted(latin_chars))}
    idx_to_latin = {idx: char for char, idx in latin_to_idx.items()}
    
    devanagari_to_idx = {char: idx for idx, char in enumerate(sorted(devanagari_chars))}
    idx_to_devanagari = {idx: char for char, idx in devanagari_to_idx.items()}
    
    return latin_to_idx, idx_to_latin, devanagari_to_idx, idx_to_devanagari

def encode_data(data, latin_to_idx, devanagari_to_idx, max_latin_len=None, max_devanagari_len=None):
    """Encode data using vocabulary mappings"""
    if max_latin_len is None:
        max_latin_len = max(len(latin) for latin, _ in data) + 2  # +2 for <SOS> and <EOS>
    if max_devanagari_len is None:
        max_devanagari_len = max(len(devanagari) for _, devanagari in data) + 2
    
    latin_encoded = []
    devanagari_encoded = []
    
    for latin, devanagari in data:
        # Encode Latin
        latin_seq = [latin_to_idx['<SOS>']]
        latin_seq.extend(latin_to_idx[char] for char in latin)
        latin_seq.append(latin_to_idx['<EOS>'])
        # Pad Latin
        latin_seq.extend([latin_to_idx['<PAD>']] * (max_latin_len - len(latin_seq)))
        latin_encoded.append(latin_seq)
        
        # Encode Devanagari
        devanagari_seq = [devanagari_to_idx['<SOS>']]
        devanagari_seq.extend(devanagari_to_idx[char] for char in devanagari)
        devanagari_seq.append(devanagari_to_idx['<EOS>'])
        # Pad Devanagari
        devanagari_seq.extend([devanagari_to_idx['<PAD>']] * (max_devanagari_len - len(devanagari_seq)))
        devanagari_encoded.append(devanagari_seq)
    
    return torch.tensor(latin_encoded), torch.tensor(devanagari_encoded), max_latin_len, max_devanagari_len

class TransliterationDataset(Dataset):
    def __init__(self, source_tensor, target_tensor):
        self.source_tensor = source_tensor
        self.target_tensor = target_tensor
    
    def __len__(self):
        return len(self.source_tensor)
    
    def __getitem__(self, idx):
        return self.source_tensor[idx], self.target_tensor[idx]
