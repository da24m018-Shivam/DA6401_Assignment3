# DA6401 Assignment 3



# Vanilla

## Directory Structure

```
.
├── main.py                  # Main entry point script
├── README.md                # This file
└── src/                     # Source code directory
    ├── data_utils.py        # Data loading and preprocessing utilities
    ├── model.py             # Neural network model definitions
    ├── test.py              # Model testing and evaluation code
    ├── train.py             # Training loop and optimization
    └── visualize.py         # Visualization utilities for model analysis
```



This project is designed to work with the [Dakshina dataset](https://github.com/google-research-datasets/dakshina), particularly the Hindi-English transliteration pairs. The expected data files are:
- `hi.translit.sampled.train.tsv`
- `hi.translit.sampled.dev.tsv`
- `hi.translit.sampled.test.tsv`

## Getting Started


### Running the Code

The main entry point is `main.py`, which supports different modes of operation:

```bash
# Train a new model
python main.py --mode train --data_dir path/to/data --model_dir models/

# Test a trained model
python main.py --mode test --data_dir path/to/data --model_dir models/

# Generate visualizations
python main.py --mode visualize --data_dir path/to/data --model_dir models/ --viz_dir visualizations/

# Run all modes sequentially
python main.py --mode all --data_dir path/to/data --model_dir models/ --viz_dir visualizations/
```

### Command Line Arguments

#### Mode Selection
- `--mode`: Operation mode (`train`, `test`, `visualize`, or `all`)

#### Directory Paths
- `--data_dir`: Directory containing the data files (default: `data/dakshina_dataset_v1.0/hi/lexicons/`)
- `--model_dir`: Directory to save/load model files (default: `models/`)
- `--viz_dir`: Directory to save visualizations (default: `visualizations/`)


## Model Architecture

The transliteration system uses a sequence-to-sequence architecture:

1. **Encoder**: Processes input Latin characters and creates a context representation
2. **Decoder**: Generates Devanagari characters based on the encoded context

The implementation supports:
- Basic RNN cells
- LSTM cells (better for handling longer sequences)
- GRU cells (computationally efficient alternative to LSTMs)

## Visualizations

The system can generate two types of visualizations:

1. **Connectivity Visualization**: Shows which input characters influence each output character
2. **Activation Visualization**: Displays the hidden state activations for both encoder and decoder



## Example Usage

Train a 2-layer LSTM model with hidden size 512:

```bash
python main.py --mode train --cell_type LSTM --hidden_size 512 --encoder_layers 2 --decoder_layers 2 --epochs 20
```

Test the model and analyze errors:

```bash
python main.py --mode test
```

Generate visualizations for the first 10 test examples:

```bash
python main.py --mode visualize --num_examples 10
```
# Attention

## Project Structure

```
.
├── src/
│   ├── config.py         # Configuration parameters and complexity calculation
│   ├── data_utils.py     # Data loading and preprocessing utilities
│   ├── model.py          # Model architecture definition
│   ├── train.py          # Training loop and related functions
│   ├── test.py           # Evaluation and error analysis
│   └── visualize.py      # Visualization utilities for attention and mappings
├── main.py               # Main entry point script
            
```

## Usage

### Basic Usage

```bash
python main.py --mode both
```

This will train and evaluate the model with default parameters.

### Advanced Options

```bash
python main.py --cell_type GRU --embed_size 256 --hidden_size 512 \
               --encoder_layers 2 --decoder_layers 2 --dropout 0.2 \
               --attention_type bahdanau --batch_size 128 \
               --learning_rate 0.001 --epochs 30
```
