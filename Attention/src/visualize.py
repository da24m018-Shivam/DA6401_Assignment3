import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
import os
from matplotlib.figure import Figure

# Configure font for proper display of Devanagari
plt.rcParams['font.family'] = 'Mangal'

def render_attention_heatmap(attention_matrix, source_text, predicted_text, target_text, output_path=None):
    """Create a visualization of attention weights"""
    fig = Figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    
    # Convert attention to numpy if it's a tensor
    if isinstance(attention_matrix, tf.Tensor):
        attention_matrix = attention_matrix.numpy()
    
    # Create heatmap
    sns.heatmap(attention_matrix, ax=ax, cmap='viridis', cbar=True)
    
    # Set axis labels
    ax.set_xticklabels([''] + list(source_text), rotation=90)
    ax.set_yticklabels([''] + list(predicted_text))
    
    # Set title
    ax.set_title(f"Target: {target_text}")
    
    # Adjust layout
    fig.tight_layout()
    
    # Save if path is provided
    if output_path:
        fig.savefig(output_path)
    
    return fig

def create_attention_visualization_grid(prediction_results, sample_count=10, output_path='results/attention_grid.png'):
    """Create a grid of attention visualizations"""
    # Limit to available examples
    sample_count = min(sample_count, len(prediction_results))
    
    # Calculate grid dimensions
    grid_rows = 5
    grid_cols = 2
    
    # Create figure
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols*4, grid_rows*4))
    
    for i, ax in enumerate(axes.flat):
        if i < sample_count:
            pred = prediction_results[i]
            
            # Get attention matrix
            attention = pred['attention']
            if isinstance(attention, tf.Tensor):
                attention = attention.numpy()
            
            # Plot heatmap
            sns.heatmap(attention, ax=ax, cmap='Blues', cbar=False)
            
            # Set tick positions and labels
            source_chars = [''] + list(pred['source'])
            prediction_chars = [''] + list(pred['prediction'])
            
            # Set x-ticks positions
            ax.set_xticks(np.arange(len(source_chars)))
            ax.set_xticklabels(source_chars, rotation=90)
            
            # Set y-ticks positions
            ax.set_yticks(np.arange(len(prediction_chars)))
            ax.set_yticklabels(prediction_chars)
            
            # Set title
            ax.set_title(f"Target: {pred['target']}", fontsize=10)
        else:
            ax.axis('off')
    
    plt.tight_layout()
    
    # Create directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save figure
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    return output_path

def visualize_character_mappings(model, data_loaders, sample_count=5, output_dir='results/character_mappings'):
    """Visualize how characters are mapped between source and target"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get a batch of examples
    for source_batch, target_batch in data_loaders['test_dataset'].take(1):
        break
    
    # Process a few examples
    for example_idx in range(min(sample_count, tf.shape(source_batch)[0])):
        # Get single example
        source_example = tf.expand_dims(source_batch[example_idx], 0)
        
        # Translate
        translations, attention_weights = model.translate(source_example)
        
        # Extract source sequence
        source_chars = []
        for i in range(tf.shape(source_example)[1]):
            idx = source_example[0, i].numpy()
            if idx in data_loaders['idx_to_roman'] and data_loaders['idx_to_roman'][idx] not in ['<start>', '<end>', '<pad>']:
                source_chars.append(data_loaders['idx_to_roman'][idx])
        
        # Extract prediction
        prediction_chars = []
        for i in range(tf.shape(translations)[1]):
            idx = translations[0, i].numpy()
            if idx in data_loaders['idx_to_native'] and data_loaders['idx_to_native'][idx] not in ['<start>', '<end>', '<pad>']:
                prediction_chars.append(data_loaders['idx_to_native'][idx])
                if idx == data_loaders['native_to_idx']['<end>']:
                    break
        
        # Create visualization
        if len(prediction_chars) > 0 and len(source_chars) > 0:
            attention_matrix = attention_weights[0, :len(prediction_chars), :len(source_chars)].numpy()
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(attention_matrix, cmap='viridis', cbar=True)
            
            plt.xticks(np.arange(len(source_chars)) + 0.5, source_chars, rotation=90)
            plt.yticks(np.arange(len(prediction_chars)) + 0.5, prediction_chars)
            
            plt.title(f"Character Mapping: {''.join(source_chars)} → {''.join(prediction_chars)}")
            plt.tight_layout()
            
            plt.savefig(f"{output_dir}/mapping_{example_idx}.png", dpi=300)
            plt.close()
    
    print(f"Character mapping visualizations saved to {output_dir}")
