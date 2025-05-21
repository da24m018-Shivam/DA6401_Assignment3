import tensorflow as tf
import time
import wandb
from tqdm import tqdm

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate):
        super(CustomSchedule, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        
    def __call__(self, step):
        return self.initial_learning_rate

def train_step(model, source, target, optimizer, loss_function, clip_norm):
    with tf.GradientTape() as tape:
        # Forward pass
        predictions = model((source, target), training=True)
        
        # Get actual batch size from the current batch
        batch_size = tf.shape(source)[0]
        
        # Reshape for loss calculation
        target_seq = target[:, 1:]  # Exclude start token
        
        # Calculate loss
        loss = loss_function(target_seq, predictions)
        
    # Calculate gradients
    gradients = tape.gradient(loss, model.trainable_variables)
    
    # Clip gradients
    gradients, _ = tf.clip_by_global_norm(gradients, clip_norm)
    
    # Apply gradients
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return tf.reduce_mean(loss)  # Return scalar loss


  # Shape: [batch_size, seq_len]



def validation_step(model, source, target, loss_function):
    """Evaluate model on validation data"""
    # Forward pass (no teacher forcing)
    predictions = model((source, target), training=False)
    
    # Reshape for loss calculation
    target_seq = target[:, 1:]  # Exclude start token
    predictions_seq = predictions  # Already aligned in model
    
    # Calculate loss
    mask = tf.math.logical_not(tf.math.equal(target_seq, 0))  # 0 is pad token
    mask = tf.cast(mask, dtype=tf.float32)
    
    loss = loss_function(target_seq, predictions_seq, sample_weight=mask)
    
    return loss

def compute_accuracy(model, dataset, target_to_idx, idx_to_target, idx_to_source):
    """Calculate word-level accuracy"""
    correct_predictions = 0
    total_samples = 0
    prediction_details = []
    
    # Find special tokens
    start_token_idx = target_to_idx['<start>']
    end_token_idx = target_to_idx['<end>']
    pad_token_idx = target_to_idx['<pad>']
    
    for source_batch, target_batch in tqdm(dataset, desc="Computing accuracy"):
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
            prediction_details.append({
                'source': source_word,
                'prediction': pred_word,
                'target': target_word,
                'attention': attn_matrix
            })
            
            # Update accuracy
            if pred_word == target_word:
                correct_predictions += 1
            total_samples += 1
    
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0
    return accuracy, prediction_details

def train_model(model, data_loaders, config):
    # Setup optimizer and loss function
    optimizer = tf.keras.optimizers.Adam(config['learning_rate'])
    loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    # Training loop
    for epoch in range(config['epochs']):
        total_loss = 0.0
        num_batches = 0
        
        for source_batch, target_batch in tqdm(data_loaders['train_dataset'], desc=f"Epoch {epoch+1} Training"):
            # Get loss for this batch
            batch_loss = train_step(
                model, source_batch, target_batch, optimizer, loss_function, config['clip_norm']
            )
            
            # Accumulate loss (use scalar value)
            total_loss += float(batch_loss)
            num_batches += 1
        
        # Calculate average loss
        train_loss = total_loss / num_batches

        
        # Rest of training loop...

 
        
        # Validation phase
        val_loss = 0
        val_steps = 0
        
        for source_batch, target_batch in tqdm(data_loaders['val_dataset'], desc="Validation"):
            batch_loss = validation_step(
                model, 
                source_batch, 
                target_batch, 
                loss_function
            )
            val_loss += batch_loss
            val_steps += 1
        
        val_loss = val_loss / val_steps
        
        # Calculate accuracy
        accuracy, _ = compute_accuracy(
            model,
            data_loaders['val_dataset'],
            data_loaders['native_to_idx'],
            data_loaders['idx_to_native'],
            data_loaders['idx_to_roman']
        )
        
        # Calculate elapsed time
        end_time = time.time()
        #minutes, seconds = divmod(end_time - start_time, 60)
        
        # Log metrics
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': accuracy,
        })
        
        # Print progress
        #print(f'Epoch: {epoch+1:02} | Time: {int(minutes)}m {seconds:.2f}s')
        print(f'\tTraining Loss: {train_loss:.3f}')
        print(f'\tValidation Loss: {val_loss:.3f}')
        print(f'\tAccuracy: {accuracy:.4f}')
        best_accuracy=0
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            model.save_weights('models/best-transliteration-model.weights.h5')
            print(f'\tNew best accuracy: {best_accuracy:.4f}, model saved')
    
    return best_accuracy
