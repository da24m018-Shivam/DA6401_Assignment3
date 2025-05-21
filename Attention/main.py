import os
import tensorflow as tf
import wandb
import argparse
import numpy as np
import random

from src.data_utils import prepare_data_pipeline
from src.model import create_model
from src.train import train_model
from src.test import evaluate_model
from src.visualize import create_attention_visualization_grid, visualize_character_mappings
from src.config import DEFAULT_CONFIG, calculate_model_complexity

def set_random_seeds(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def main(args):
    # Set random seed
    set_random_seeds(args.seed)
    
    # Check for GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Using {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(e)
    else:
        print("Using CPU")
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Initialize wandb
    wandb.login(key=args.wandb_key)
    
    # Create config dictionary
    config = {
        'cell_type': args.cell_type,
        'embed_size': args.embed_size,
        'hidden_size': args.hidden_size,
        'encoder_layers': args.encoder_layers,
        'decoder_layers': args.decoder_layers,
        'dropout': args.dropout,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'teacher_forcing_ratio': args.teacher_forcing_ratio,
        'attention_type': args.attention_type,
        'clip_norm': args.clip_norm,
        'epochs': args.epochs
    }
    
    # Initialize wandb run
    wandb.init(
        project="tf-transliteration-attention",
        config=config
    )
    
    # Prepare data
    data_loaders = prepare_data_pipeline(args.data_dir, batch_size=args.batch_size)
    
    # Initialize model
    source_vocab_size = len(data_loaders['roman_to_idx'])
    target_vocab_size = len(data_loaders['native_to_idx'])
    
    # Get special token indices
    # Get special token indices
    start_token_idx = data_loaders['native_to_idx']['<start>']
    end_token_idx = data_loaders['native_to_idx']['<end>']

    model = create_model(
        source_vocab_size=source_vocab_size,
        target_vocab_size=target_vocab_size,
        embedding_dim=args.embed_size,
        hidden_units=args.hidden_size,
        cell_type=args.cell_type,
        num_layers=args.encoder_layers,
        dropout_rate=args.dropout,
        attention_type=args.attention_type,
        start_token_idx=start_token_idx,
        end_token_idx=end_token_idx
    )

    
    # Calculate model complexity
    complexity = calculate_model_complexity(
        embedding_size=args.embed_size,
        hidden_size=args.hidden_size,
        sequence_length=max(data_loaders['max_roman_len'], data_loaders['max_native_len']),
        vocabulary_size=max(source_vocab_size, target_vocab_size)
    )
    
    # Log model complexity
    wandb.log(complexity)
    
    # Train model
    if args.mode in ['train', 'both']:
        print("Starting training...")
        train_model(model, data_loaders, config)
    
    # Test model
    if args.mode in ['test', 'both']:
        print("Starting testing...")
        _, predictions = evaluate_model(model, data_loaders)
        
        # Create attention visualizations
        print("Creating attention visualizations...")
        create_attention_visualization_grid(predictions, sample_count=10, output_path='results/attention_grid.png')
        
        # Create character mapping visualizations
        print("Creating character mapping visualizations...")
        visualize_character_mappings(model, data_loaders, sample_count=5)
        
        # Log visualizations to wandb
        wandb.log({"attention_grid": wandb.Image('results/attention_grid.png')})
        
        # Log character mapping visualizations
        for i in range(5):
            if os.path.exists(f'results/character_mappings/mapping_{i}.png'):
                wandb.log({f"character_mapping_{i}": wandb.Image(f'results/character_mappings/mapping_{i}.png')})
    
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TensorFlow Transliteration with Attention")
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data/dakshina_dataset_v1.0/hi/lexicons',
                        help='Path to the data directory')
    
    # Model arguments
    parser.add_argument('--cell_type', type=str, default=DEFAULT_CONFIG['cell_type'],
                        choices=['RNN', 'LSTM', 'GRU'], help='Type of RNN cell')
    parser.add_argument('--embed_size', type=int, default=DEFAULT_CONFIG['embed_size'],
                        help='Embedding size')
    parser.add_argument('--hidden_size', type=int, default=DEFAULT_CONFIG['hidden_size'],
                        help='Hidden size')
    parser.add_argument('--encoder_layers', type=int, default=DEFAULT_CONFIG['encoder_layers'],
                        help='Number of encoder layers')
    parser.add_argument('--decoder_layers', type=int, default=DEFAULT_CONFIG['decoder_layers'],
                        help='Number of decoder layers')
    parser.add_argument('--dropout', type=float, default=DEFAULT_CONFIG['dropout'],
                        help='Dropout rate')
    parser.add_argument('--attention_type', type=str, default=DEFAULT_CONFIG['attention_type'],
                        choices=['luong', 'bahdanau'], help='Type of attention mechanism')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=DEFAULT_CONFIG['batch_size'],
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=DEFAULT_CONFIG['learning_rate'],
                        help='Learning rate')
    parser.add_argument('--teacher_forcing_ratio', type=float, default=DEFAULT_CONFIG['teacher_forcing_ratio'],
                        help='Teacher forcing ratio')
    parser.add_argument('--clip_norm', type=float, default=DEFAULT_CONFIG['clip_norm'],
                        help='Gradient clipping value')
    parser.add_argument('--epochs', type=int, default=DEFAULT_CONFIG['epochs'],
                        help='Number of epochs')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--wandb_key', type=str, default="93b4881869bab13360839595daa56e51dd0405df", help='WandB API key')
    parser.add_argument('--mode', type=str, default='both', choices=['train', 'test', 'both'],
                        help='Mode to run the script')
    
    args = parser.parse_args()
    main(args)
