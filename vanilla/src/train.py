import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import wandb

def train(model, iterator, optimizer, criterion, clip, teacher_forcing_ratio, device):
    model.train()
    epoch_loss = 0
    
    for i, (src, trg) in enumerate(iterator):
        src, trg = src.to(device), trg.to(device)
        
        optimizer.zero_grad()
        
        output = model(src, trg, teacher_forcing_ratio)
        
        # Reshape output and target for loss calculation
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)
        
        loss = criterion(output, trg)
        loss.backward()
        
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion, device):
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for i, (src, trg) in enumerate(iterator):
            src, trg = src.to(device), trg.to(device)
            
            output = model(src, trg, 0)  # No teacher forcing during evaluation
            
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)
            
            loss = criterion(output, trg)
            
            epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)

def train_model(config, data_dir, model_dir, device, log_wandb=False):
    """Train the model with the given configuration"""
    from src.data_utils import load_data, create_vocab, encode_data, TransliterationDataset
    from src.model import EncoderRNN, DecoderRNN, Seq2Seq
    
    # Load data
    train_data = load_data(os.path.join(data_dir, 'hi.translit.sampled.train.tsv'))
    val_data = load_data(os.path.join(data_dir, 'hi.translit.sampled.dev.tsv'))
    
    # Create vocabulary
    latin_to_idx, idx_to_latin, devanagari_to_idx, idx_to_devanagari = create_vocab(train_data + val_data)
    
    # Encode data
    X_train, y_train, max_latin_len, max_devanagari_len = encode_data(train_data, latin_to_idx, devanagari_to_idx)
    X_val, y_val, _, _ = encode_data(val_data, latin_to_idx, devanagari_to_idx, max_latin_len, max_devanagari_len)
    
    # Create datasets and dataloaders
    train_dataset = TransliterationDataset(X_train, y_train)
    val_dataset = TransliterationDataset(X_val, y_val)
    
    train_iterator = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_iterator = torch.utils.data.DataLoader(val_dataset, batch_size=config['batch_size'])
    
    # Initialize model
    input_size = len(latin_to_idx)
    output_size = len(devanagari_to_idx)
    
    encoder = EncoderRNN(
        input_size=input_size,
        embed_size=config['embed_size'],
        hidden_size=config['hidden_size'],
        cell_type=config['cell_type'],
        num_layers=config['encoder_layers'],
        dropout=config['dropout']
    )
    
    decoder = DecoderRNN(
        output_size=output_size,
        embed_size=config['embed_size'],
        hidden_size=config['hidden_size'],
        cell_type=config['cell_type'],
        num_layers=config['decoder_layers'],
        dropout=config['dropout']
    )
    
    model = Seq2Seq(encoder, decoder).to(device)
    
    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss(ignore_index=devanagari_to_idx['<PAD>'])
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(config['epochs']):
        start_time = time.time()
        
        train_loss = train(model, train_iterator, optimizer, criterion, config['clip'], config['teacher_forcing_ratio'], device)
        val_loss = evaluate(model, val_iterator, criterion, device)
        
        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
        
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs:.2f}s')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\tVal Loss: {val_loss:.3f}')
        
        # Log to wandb if enabled
        if log_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'time_mins': epoch_mins,
                'time_secs': epoch_secs
            })
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(model_dir, 'best-model.pt'))
            if log_wandb:
                wandb.run.summary['best_val_loss'] = best_val_loss
    
    # Save vocabulary for later use
    torch.save({
        'latin_to_idx': latin_to_idx,
        'idx_to_latin': idx_to_latin,
        'devanagari_to_idx': devanagari_to_idx,
        'idx_to_devanagari': idx_to_devanagari,
        'max_latin_len': max_latin_len,
        'max_devanagari_len': max_devanagari_len
    }, os.path.join(model_dir, 'vocab.pt'))
    
    return model, latin_to_idx, idx_to_latin, devanagari_to_idx, idx_to_devanagari
