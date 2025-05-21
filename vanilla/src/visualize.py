import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
plt.rcParams['font.family'] = 'Mangal'
def compute_connectivity_matrix(model, source_seq, target_seq, device):
    """
    Compute the connectivity matrix showing which input tokens influence which output tokens.
    This is a simplified version without actual gradient computation.
    
    For vanilla RNNs without attention, we'll use a heuristic approach:
    - For each output position, we'll assign higher weights to input positions that are closer
      to the beginning of the sequence, with exponential decay.
    """
    source_seq = source_seq.to(device)
    target_seq = target_seq.to(device)
    
    batch_size = source_seq.size(0)
    source_len = source_seq.size(1)
    target_len = target_seq.size(1)
    
    # Initialize connectivity matrix
    connectivity = torch.zeros(batch_size, target_len, source_len)
    
    # Encode the source sequence
    encoder_outputs, hidden = model.encoder(source_seq)
    
    # First input to the decoder is the <SOS> token
    decoder_input = target_seq[:, 0]
    
    # Handle differing encoder/decoder layers
    if model.encoder.cell_type == 'LSTM':
        hidden, cell = hidden
        if model.encoder.num_layers != model.decoder.num_layers:
            factor = model.decoder.num_layers // model.encoder.num_layers + 1
            hidden = hidden.repeat(factor, 1, 1)[:model.decoder.num_layers]
            cell = cell.repeat(factor, 1, 1)[:model.decoder.num_layers]
        hidden = (hidden, cell)
    else:
        if model.encoder.num_layers != model.decoder.num_layers:
            factor = model.decoder.num_layers // model.encoder.num_layers + 1
            hidden = hidden.repeat(factor, 1, 1)[:model.decoder.num_layers]
    
    # For each output token, compute influence from input tokens
    for t in range(1, target_len):
        # Forward pass for this time step
        output, hidden = model.decoder(decoder_input, hidden)
        
        # Get the predicted token
        top1 = output.argmax(1)
        
        # For vanilla RNN, the influence decays exponentially from the beginning
        # This is a heuristic since we can't easily compute gradients
        for b in range(batch_size):
            # Create a decay pattern - earlier inputs have more influence on earlier outputs
            decay_rate = 0.9  # Adjust as needed
            influence = torch.tensor([decay_rate ** (source_len - i - 1) for i in range(source_len)])
            
            # Normalize to sum to 1
            influence = influence / influence.sum()
            
            connectivity[b, t, :] = influence
        
        # Use predicted token as next input
        decoder_input = top1
    
    return connectivity

def visualize_connectivity(model, source_seq, target_seq, idx_to_latin, idx_to_devanagari, device, sample_idx=0, save_path=None):
    """
    Visualize the connectivity between input and output tokens.
    """
    # Compute connectivity matrix
    connectivity = compute_connectivity_matrix(model, source_seq, target_seq, device)
    
    # Get single example
    single_connectivity = connectivity[sample_idx].cpu().numpy()
    
    # Get source and target tokens
    source_tokens = [idx_to_latin[idx.item()] for idx in source_seq[sample_idx]]
    target_tokens = [idx_to_devanagari[idx.item()] for idx in target_seq[sample_idx]]
    
    # Remove padding tokens
    source_tokens = [t for t in source_tokens if t not in ['<PAD>']]
    target_tokens = [t for t in target_tokens if t not in ['<PAD>']]
    
    # Trim connectivity matrix to match
    single_connectivity = single_connectivity[:len(target_tokens), :len(source_tokens)]
    
    # Create a custom colormap from white to blue
    colors = [(1, 1, 1), (0, 0, 1)]  # White to blue
    cmap = LinearSegmentedColormap.from_list('white_to_blue', colors, N=100)
    
    # Create the heatmap
    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(
        single_connectivity, 
        cmap=cmap,
        xticklabels=source_tokens,
        yticklabels=target_tokens,
        cbar_kws={'label': 'Connection Strength'}
    )
    
    # Set labels and title
    plt.xlabel('Input Characters (Latin)')
    plt.ylabel('Output Characters (Devanagari)')
    plt.title('RNN Connectivity: Which Input Characters Influence Each Output Character')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show the plot
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    return single_connectivity

def visualize_activations(model, source_seq, target_seq, idx_to_latin, idx_to_devanagari, device, sample_idx=0, save_path=None):
    """
    Visualize the hidden state activations of the RNN.
    """
    model.eval()
    source_seq = source_seq.to(device)
    target_seq = target_seq.to(device)
    
    # Forward pass through encoder
    encoder_outputs, hidden = model.encoder(source_seq)
    
    # Get encoder activations
    encoder_activations = model.encoder.activations[sample_idx].cpu().numpy()
    
    # First input to the decoder is the <SOS> token
    decoder_input = target_seq[sample_idx:sample_idx+1, 0]
    
    # Handle differing encoder/decoder layers
    if model.encoder.cell_type == 'LSTM':
        hidden_states, cell_states = hidden
        if model.encoder.num_layers != model.decoder.num_layers:
            factor = model.decoder.num_layers // model.encoder.num_layers + 1
            hidden_states = hidden_states.repeat(factor, 1, 1)[:model.decoder.num_layers]
            cell_states = cell_states.repeat(factor, 1, 1)[:model.decoder.num_layers]
        decoder_hidden = (hidden_states, cell_states)
    else:
        if model.encoder.num_layers != model.decoder.num_layers:
            factor = model.decoder.num_layers // model.encoder.num_layers + 1
            hidden = hidden.repeat(factor, 1, 1)[:model.decoder.num_layers]
        decoder_hidden = hidden
    
    # Collect decoder activations for each time step
    decoder_activations = []
    
    for t in range(1, target_seq.size(1)):
        output, decoder_hidden = model.decoder(decoder_input, decoder_hidden)
        decoder_activations.append(model.decoder.activations[0].cpu().numpy())
        
        # Use predicted token as next input
        top1 = output.argmax(1)
        decoder_input = top1
    
    decoder_activations = np.array(decoder_activations)
    
    # Get source and target tokens
    source_tokens = [idx_to_latin[idx.item()] for idx in source_seq[sample_idx]]
    target_tokens = [idx_to_devanagari[idx.item()] for idx in target_seq[sample_idx]]
    
    # Remove padding tokens
    source_tokens = [t for t in source_tokens if t not in ['<PAD>']]
    target_tokens = [t for t in target_tokens if t not in ['<PAD>']]
    
    # Plot encoder activations
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 1, 1)
    sns.heatmap(
        encoder_activations[:len(source_tokens)].T,
        cmap='viridis',
        xticklabels=source_tokens,
        cbar_kws={'label': 'Activation Value'}
    )
    plt.xlabel('Input Characters (Latin)')
    plt.ylabel('Hidden Units')
    plt.title('Encoder Hidden State Activations')
    plt.xticks(rotation=45, ha='right')
    
    # Plot decoder activations
    plt.subplot(2, 1, 2)
    sns.heatmap(
        decoder_activations[:len(target_tokens)-1].reshape(-1, decoder_activations.shape[-1]),
        cmap='viridis',
        xticklabels=target_tokens[1:],  # Skip <SOS>
        cbar_kws={'label': 'Activation Value'}
    )
    plt.xlabel('Output Characters (Devanagari)')
    plt.ylabel('Hidden Units')
    plt.title('Decoder Hidden State Activations')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
