import tensorflow as tf
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from tqdm import tqdm
from collections import Counter
import pandas as pd
# Set font for Devanagari display
plt.rcParams['font.family'] = 'Mangal'

def evaluate_model(model, data_loaders):
    """Evaluate model on test set and analyze results"""
    # Load best model weights
    model.load_weights('models/best-transliteration-model.weights.h5')
    
    # Calculate accuracy and get predictions
    test_accuracy, prediction_results = compute_test_accuracy(
        model,
        data_loaders['test_dataset'],
        data_loaders['native_to_idx'],
        data_loaders['idx_to_native'],
        data_loaders['idx_to_roman']
    )
    
    # Calculate character-level accuracy
    char_accuracy = compute_character_accuracy(prediction_results)
    
    # Analyze error patterns
    error_patterns = analyze_error_patterns(prediction_results)
    
    # Log metrics to wandb
    wandb.log({
        'test_word_accuracy': test_accuracy,
        'test_char_accuracy': char_accuracy
    })
    
    # Print results
    print(f'Test Word Accuracy: {test_accuracy:.4f}')
    print(f'Test Character Accuracy: {char_accuracy:.4f}')
    
    # Print most common errors
    print("\nMost common error patterns:")
    for (target_char, pred_char), count in sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"Target '{target_char}' predicted as '{pred_char}': {count} times")
    
    # Log sample predictions
    sample_entries = []
    for i, pred in enumerate(prediction_results[:10]):
        sample_entries.append([i, pred['prediction'], pred['target'], pred['prediction'] == pred['target']])
        df = pd.DataFrame([
        {
            'Index': i,
            'Prediction': pred['prediction'],
            'Target': pred['target'],
            'Correct': pred['prediction'] == pred['target']
        }
        for i, pred in enumerate(prediction_results)
    ])

    # Save to CSV
    df.to_csv('prediction_results.csv', index=False)
    
    wandb.log({
        "prediction_samples": wandb.Table(
            columns=["Index", "Prediction", "Target", "Correct"],
            data=sample_entries
        )
    })
    
    # Create error analysis visualization
    top_errors = sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)[:10]
    error_labels = [f"{t}->{p}" for (t, p), _ in top_errors]
    error_counts = [count for _, count in top_errors]
    
    plt.figure(figsize=(10, 6))
    plt.bar(error_labels, error_counts)
    plt.xlabel('Error Type (Target->Prediction)')
    plt.ylabel('Count')
    plt.title('Top 10 Error Patterns')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Log visualization
    wandb.log({"error_analysis": wandb.Image(plt)})
    plt.close()
    
    return test_accuracy, prediction_results

def compute_test_accuracy(model, dataset, target_to_idx, idx_to_target, idx_to_source):
    """Calculate accuracy on test set and collect detailed predictions"""
    correct_count = 0
    total_count = 0
    all_predictions = []
    
    # Find special tokens
    start_token_idx = target_to_idx['<start>']
    end_token_idx = target_to_idx['<end>']
    pad_token_idx = target_to_idx['<pad>']
    
    for source_batch, target_batch in tqdm(dataset, desc="Testing"):
        batch_size = tf.shape(source_batch)[0]
        
        # Translate
        translations, attention_weights = model.translate(source_batch)
        
        # Process each example in batch
        for i in range(batch_size):
            # Extract source sequence
            source_seq = []
            for j in range(tf.shape(source_batch)[1]):
                idx = source_batch[i, j].numpy()
                if idx in idx_to_source and idx_to_source[idx] not in ['<start>', '<end>', '<pad>']:
                    source_seq.append(idx_to_source[idx])
            
            # Extract prediction and target sequences
            pred_seq = []
            target_seq = []
            
            # Process prediction
            for j in range(tf.shape(translations)[1]):
                pred_idx = translations[i, j].numpy()
                if pred_idx == end_token_idx or pred_idx == pad_token_idx:
                    break
                if pred_idx in idx_to_target and idx_to_target[pred_idx] not in ['<start>', '<end>', '<pad>']:
                    pred_seq.append(idx_to_target[pred_idx])
            
            # Process target
            for j in range(1, tf.shape(target_batch)[1]):  # Skip start token
                target_idx = target_batch[i, j].numpy()
                if target_idx == end_token_idx or target_idx == pad_token_idx:
                    break
                if target_idx in idx_to_target and idx_to_target[target_idx] not in ['<start>', '<end>', '<pad>']:
                    target_seq.append(idx_to_target[target_idx])
            
            # Convert to strings
            pred_word = ''.join(pred_seq)
            target_word = ''.join(target_seq)
            source_word = ''.join(source_seq)
            
            # Get attention weights
            attn_matrix = attention_weights[i, :len(pred_seq), :len(source_seq)].numpy()
            
            # Store prediction details
            all_predictions.append({
                'source': source_word,
                'prediction': pred_word,
                'target': target_word,
                'attention': attn_matrix
            })
            
            # Update accuracy counters
            if pred_word == target_word:
                correct_count += 1
            total_count += 1
    
    accuracy = correct_count / total_count if total_count > 0 else 0
    return accuracy, all_predictions

def compute_character_accuracy(predictions):
    """Calculate character-level accuracy"""
    total_chars = 0
    correct_chars = 0
    
    for pred in predictions:
        prediction = pred['prediction']
        target = pred['target']
        
        # Compare characters up to the length of the shorter string
        min_len = min(len(prediction), len(target))
        for i in range(min_len):
            if prediction[i] == target[i]:
                correct_chars += 1
            total_chars += 1
        
        # Count remaining characters in longer string as errors
        total_chars += abs(len(prediction) - len(target))
    
    char_accuracy = correct_chars / total_chars if total_chars > 0 else 0
    return char_accuracy

def analyze_error_patterns(predictions):
    """Analyze common error patterns in predictions"""
    error_counter = Counter()
    
    for pred in predictions:
        prediction = pred['prediction']
        target = pred['target']
        
        if prediction != target:
            # Find the first position where they differ
            min_len = min(len(prediction), len(target))
            for i in range(min_len):
                if prediction[i] != target[i]:
                    error_pair = (target[i], prediction[i])
                    error_counter[error_pair] += 1
                    break
            
            # Handle length differences
            if len(prediction) != len(target) and min_len > 0 and i == min_len - 1:
                if len(prediction) < len(target):
                    error_pair = (target[min_len], "MISSING")
                else:
                    error_pair = ("MISSING", prediction[min_len])
                error_counter[error_pair] += 1
    
    return error_counter
