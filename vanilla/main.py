import os
import torch
import argparse
import wandb
from src.model import EncoderRNN, DecoderRNN, Seq2Seq
from src.train import train_model
from src.test import test_model
from src.visualize import visualize_connectivity, visualize_activations
from src.data_utils import load_data, encode_data, TransliterationDataset

def main():
    parser = argparse.ArgumentParser(description='Train, test, and visualize RNN for transliteration')
    
    # Mode selection
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'visualize', 'all'], 
                        help='Mode to run: train, test, visualize, or all')
    
    # Data and model paths
    parser.add_argument('--data_dir', type=str, default='data/dakshina_dataset_v1.0/hi/lexicons/',
                        help='Directory containing the data files')
    parser.add_argument('--model_dir', type=str, default='models/',
                        help='Directory to save/load model files')
    parser.add_argument('--viz_dir', type=str, default='visualizations/',
                        help='Directory to save visualizations locally')
    
    # Model architecture parameters
    parser.add_argument('--cell_type', type=str, default='LSTM', choices=['RNN', 'LSTM', 'GRU'],
                        help='Type of RNN cell to use')
    parser.add_argument('--embed_size', type=int, default=128,
                        help='Size of embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='Size of hidden state')
    parser.add_argument('--encoder_layers', type=int, default=1,
                        help='Number of encoder layers')
    parser.add_argument('--decoder_layers', type=int, default=1,
                        help='Number of decoder layers')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--teacher_forcing_ratio', type=float, default=0.5,
                        help='Teacher forcing ratio')
    parser.add_argument('--clip', type=float, default=1.0,
                        help='Gradient clipping value')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    
    # Visualization parameters
    parser.add_argument('--num_examples', type=int, default=5,
                        help='Number of examples to visualize')
    
    # Weights & Biases parameters
    parser.add_argument('--log_wandb', action='store_true',
                        help='Whether to log results to wandb')
    parser.add_argument('--project_name', type=str, default='transliteration-attention_final_final_2',
                        help='Project name for wandb')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='Entity (username or team name) for wandb')
    
    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.viz_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize wandb if logging is enabled
    if args.log_wandb:
        wandb.init(
            project=args.project_name,
            entity=args.wandb_entity,
            config=vars(args),
            name=f"{args.cell_type}_{args.hidden_size}_{args.encoder_layers}_{args.decoder_layers}"
        )
    
    # Convert args to config dictionary
    config = vars(args)
    
    # Initialize model
    if args.mode in ['test', 'visualize', 'all']:
        # Load vocabulary
        try:
            vocab_data = torch.load(os.path.join(args.model_dir, 'vocab.pt'))
            latin_to_idx = vocab_data['latin_to_idx']
            idx_to_latin = vocab_data['idx_to_latin']
            devanagari_to_idx = vocab_data['devanagari_to_idx']
            idx_to_devanagari = vocab_data['idx_to_devanagari']
            
            # Initialize model with vocabulary sizes
            encoder = EncoderRNN(
                input_size=len(latin_to_idx),
                embed_size=args.embed_size,
                hidden_size=args.hidden_size,
                cell_type=args.cell_type,
                num_layers=args.encoder_layers,
                dropout=args.dropout
            )
            
            decoder = DecoderRNN(
                output_size=len(devanagari_to_idx),
                embed_size=args.embed_size,
                hidden_size=args.hidden_size,
                cell_type=args.cell_type,
                num_layers=args.decoder_layers,
                dropout=args.dropout
            )
            
            model = Seq2Seq(encoder, decoder).to(device)
            
            # Load model weights if not training
            if args.mode != 'all':
                model.load_state_dict(torch.load(os.path.join(args.model_dir, 'best-model.pt')))
                model.eval()
        except FileNotFoundError:
            if args.mode == 'train' or args.mode == 'all':
                print("Vocabulary file not found. Will be created during training.")
            else:
                print("Error: Vocabulary file not found. Please train the model first.")
                return
    
    # Execute based on mode
    if args.mode in ['train', 'all']:
        # Train the model
        model, latin_to_idx, idx_to_latin, devanagari_to_idx, idx_to_devanagari = train_model(
            config, args.data_dir, args.model_dir, device, log_wandb=args.log_wandb
        )
        print("Training completed!")
    
    if args.mode in ['test', 'all']:
        # Test the model
        test_accuracy, char_accuracy, predictions, error_analysis = test_model(
            model, args.data_dir, args.model_dir, device, config, log_wandb=args.log_wandb
        )
        print(f"Testing completed! Word Accuracy: {test_accuracy:.4f}, Char Accuracy: {char_accuracy:.4f}")
    
    if args.mode in ['visualize', 'all']:
        # Load test data for visualization
        test_data = load_data(os.path.join(args.data_dir, 'hi.translit.sampled.test.tsv'))
        
        # Encode test data
        X_test, y_test, _, _ = encode_data(
            test_data[:args.num_examples],
            latin_to_idx, 
            devanagari_to_idx
        )
        
        # Visualize connectivity and activations for each example
        for i in range(min(args.num_examples, len(X_test))):
            # Create connectivity visualization
            connectivity_path = os.path.join(args.viz_dir, f'connectivity_example_{i}.png')
            connectivity = visualize_connectivity(
                model, 
                X_test[i:i+1], 
                y_test[i:i+1], 
                idx_to_latin, 
                idx_to_devanagari, 
                device,
                sample_idx=0,
                save_path=connectivity_path
            )
            print(f"Saved connectivity visualization to {connectivity_path}")
            
            # Create activations visualization
            activations_path = os.path.join(args.viz_dir, f'activations_example_{i}.png')
            visualize_activations(
                model, 
                X_test[i:i+1], 
                y_test[i:i+1], 
                idx_to_latin, 
                idx_to_devanagari, 
                device,
                sample_idx=0,
                save_path=activations_path
            )
            print(f"Saved activations visualization to {activations_path}")
            
            # Log visualizations to wandb
            if args.log_wandb:
                wandb.log({
                    f"connectivity_example_{i}": wandb.Image(connectivity_path),
                    f"activations_example_{i}": wandb.Image(activations_path)
                })
    
    # Finish wandb run
    if args.log_wandb:
        wandb.finish()

if __name__ == '__main__':
    main()
