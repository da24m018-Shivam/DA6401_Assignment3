import torch
import os
import wandb
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
plt.rcParams['font.family'] = 'Mangal'
def calculate_accuracy(model, iterator, devanagari_to_idx, idx_to_devanagari, device):
    """Calculate accuracy on the test set"""
    model.eval()
    correct = 0
    total = 0
    predictions = []
    
    with torch.no_grad():
        for src, trg in iterator:
            src, trg = src.to(device), trg.to(device)
            batch_size = src.shape[0]
            
            # Encode the source sequence
            _, hidden = model.encoder(src)
            
            # Start with <SOS> token
            decoder_input = torch.tensor([devanagari_to_idx['<SOS>']] * batch_size).to(device)
            
            # Store predictions
            trg_len = trg.shape[1]
            predictions_batch = torch.zeros(batch_size, trg_len).to(device)
            predictions_batch[:, 0] = decoder_input
            
            for t in range(1, trg_len):
                output, hidden = model.decoder(decoder_input, hidden)
                top1 = output.argmax(1)
                predictions_batch[:, t] = top1
                decoder_input = top1
            
            # Compare predictions with targets
            for i in range(batch_size):
                pred_seq = []
                target_seq = []
                
                for j in range(1, trg_len):  # Skip <SOS>
                    pred_idx = predictions_batch[i, j].item()
                    target_idx = trg[i, j].item()
                    
                    if pred_idx == devanagari_to_idx['<EOS>'] or pred_idx == devanagari_to_idx['<PAD>']:
                        break
                    if target_idx == devanagari_to_idx['<EOS>'] or target_idx == devanagari_to_idx['<PAD>']:
                        break
                    
                    pred_seq.append(idx_to_devanagari[pred_idx])
                    target_seq.append(idx_to_devanagari[target_idx])
                
                pred_word = ''.join(pred_seq)
                target_word = ''.join(target_seq)
                
                predictions.append((pred_word, target_word))
                
                if pred_word == target_word:
                    correct += 1
                total += 1
    
    accuracy = correct / total if total > 0 else 0
    return accuracy, predictions

def analyze_errors(predictions):
    """Analyze common error patterns in predictions"""
    error_analysis = {}
    
    for pred, target in predictions:
        if pred != target:
            # Find the first position where they differ
            min_len = min(len(pred), len(target))
            for i in range(min_len):
                if pred[i] != target[i]:
                    error_pair = (target[i], pred[i])
                    error_analysis[error_pair] = error_analysis.get(error_pair, 0) + 1
                    break
            
            # Handle length differences
            if len(pred) != len(target) and i == min_len - 1:
                if len(pred) < len(target):
                    error_pair = (target[min_len], "MISSING")
                else:
                    error_pair = ("MISSING", pred[min_len])
                error_analysis[error_pair] = error_analysis.get(error_pair, 0) + 1
    
    return error_analysis

def calculate_char_accuracy(predictions):
    """Calculate character-level accuracy"""
    total_chars = 0
    correct_chars = 0
    
    for pred, target in predictions:
        min_len = min(len(pred), len(target))
        
        for i in range(min_len):
            if pred[i] == target[i]:
                correct_chars += 1
            total_chars += 1
        
        # Count remaining characters in longer string as errors
        total_chars += abs(len(pred) - len(target))
    
    char_accuracy = correct_chars / total_chars if total_chars > 0 else 0
    return char_accuracy

def test_model(model, data_dir, model_dir, device, config, log_wandb=False, save_csv=True):
    """Test the model on the test dataset"""
    from src.data_utils import load_data, encode_data, TransliterationDataset
    import csv
    
    # Load vocabulary
    vocab_data = torch.load(os.path.join(model_dir, 'vocab.pt'))
    latin_to_idx = vocab_data['latin_to_idx']
    idx_to_latin = vocab_data['idx_to_latin']
    devanagari_to_idx = vocab_data['devanagari_to_idx']
    idx_to_devanagari = vocab_data['idx_to_devanagari']
    max_latin_len = vocab_data.get('max_latin_len', None)
    max_devanagari_len = vocab_data.get('max_devanagari_len', None)
    
    # Load test data
    test_data = load_data(os.path.join(data_dir, 'hi.translit.sampled.test.tsv'))
    
    # Encode test data
    X_test, y_test, _, _ = encode_data(
        test_data, 
        latin_to_idx, 
        devanagari_to_idx, 
        max_latin_len, 
        max_devanagari_len
    )
    
    # Create test dataset and dataloader
    test_dataset = TransliterationDataset(X_test, y_test)
    test_iterator = DataLoader(test_dataset, batch_size=config['batch_size'])
    
    # Load the best model
    model.load_state_dict(torch.load(os.path.join(model_dir, 'best-model.pt')))
    model.eval()
    
    # Calculate test accuracy
    test_accuracy, predictions = calculate_accuracy(
        model, 
        test_iterator, 
        devanagari_to_idx, 
        idx_to_devanagari, 
        device
    )
    
    # Calculate character-level accuracy
    char_accuracy = calculate_char_accuracy(predictions)
    
    # Analyze errors
    error_analysis = analyze_errors(predictions)
    
    # Print results
    print(f'Test Word Accuracy: {test_accuracy:.4f}')
    print(f'Test Character Accuracy: {char_accuracy:.4f}')
    
    # Print most common errors
    print("\nMost common errors:")
    for (target_char, pred_char), count in sorted(error_analysis.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"Target '{target_char}' predicted as '{pred_char}': {count} times")
    
    # Save predictions to CSV if enabled
    if save_csv:
        csv_path = os.path.join(model_dir, 'predictions.csv')
        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                # Write header
                writer.writerow(['Target', 'Predicted', 'Correct'])
                
                # Write prediction data
                for pred, target in predictions:
                    writer.writerow([target, pred, pred == target])
                    
            print(f"Predictions saved to {os.path.abspath(csv_path)}")
        except Exception as e:
            print(f"Error saving predictions to CSV: {e}")
    
    # Log to wandb if enabled
    if log_wandb:
        wandb.log({
            'test_word_accuracy': test_accuracy,
            'test_char_accuracy': char_accuracy
        })
        
        # Log sample predictions as a table
        samples = []
        for i, (pred, target) in enumerate(predictions[:20]):
            samples.append([i, pred, target, pred == target])
        
        wandb.log({
            "test_predictions": wandb.Table(
                columns=["Index", "Prediction", "Target", "Correct"],
                data=samples
            )
        })
        
        # Log error analysis as a bar chart
        top_errors = sorted(error_analysis.items(), key=lambda x: x[1], reverse=True)[:10]
        error_labels = [f"{t}->{p}" for (t, p), _ in top_errors]
        error_counts = [count for _, count in top_errors]
        
        plt.figure(figsize=(10, 6))
        plt.bar(error_labels, error_counts)
        plt.xlabel('Error Type (Target->Prediction)')
        plt.ylabel('Count')
        plt.title('Top 10 Error Types')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        wandb.log({"error_analysis": wandb.Image(plt)})
        plt.close()
    
    return test_accuracy, char_accuracy, predictions, error_analysis