import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Flatten, Concatenate, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
import os
# Add these imports alongside your existing imports
from load_dkt_data import load_dkt_datasets, prepare_kt_sequences

# Disable TensorFlow warnings and set memory growth
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Function to generate synthetic data for knowledge tracing with more realistic learning patterns
def generate_synthetic_data(n_users=500, n_skills=100, min_interactions=15, max_interactions=50):
    """
    Generate synthetic data for knowledge tracing with realistic learning patterns.
    """
    print("Generating synthetic data for knowledge tracing...")
    np.random.seed(42)
    
    data_rows = []
    
    # Create skill difficulties (harder skills have lower base probability)
    skill_difficulties = {}
    for skill in range(1, n_skills + 1):
        # Random difficulty between 0.3 (hard) and 0.7 (easy)
        skill_difficulties[skill] = 0.3 + (0.4 * np.random.random())
    
    # Create user abilities (higher ability = higher success probability)
    user_abilities = {}
    for user in range(1, n_users + 1):
        # Random ability between -0.2 (struggling) and 0.2 (advanced)
        user_abilities[user] = -0.2 + (0.4 * np.random.random())
    
    for user_id in range(1, n_users + 1):
        # Each user has a sequence of interactions
        n_user_interactions = np.random.randint(min_interactions, max_interactions)
        
        # Select a subset of skills for this user (realistic - students don't practice all skills)
        user_skill_count = np.random.randint(min(10, n_skills // 2), min(30, n_skills))
        available_skills = np.random.choice(range(1, n_skills + 1), user_skill_count, replace=False)
        
        # Create sequence of skills (with repetition to show learning)
        skills = []
        for _ in range(n_user_interactions):
            # 70% chance to practice a previously seen skill, 30% chance for a new one
            if skills and np.random.random() < 0.7:
                # Select from previously seen skills, with higher probability for recent ones
                skills_seen = list(set(skills))
                # Weight recent skills more heavily
                weights = np.linspace(0.5, 1.0, len(skills_seen))
                skill = np.random.choice(skills_seen, p=weights/weights.sum())
            else:
                # Pick a skill they haven't practiced yet
                unused_skills = [s for s in available_skills if s not in skills]
                if unused_skills:
                    skill = np.random.choice(unused_skills)
                else:
                    # If all skills used, pick a random one from their skill set
                    skill = np.random.choice(available_skills)
            
            skills.append(skill)
        
        # Generate correctness with more realistic learning curve
        correctness = []
        skill_attempts = {}
        for i, skill in enumerate(skills):
            if skill not in skill_attempts:
                skill_attempts[skill] = 0
            
            # Base probability from skill difficulty and user ability
            base_prob = skill_difficulties[skill] + user_abilities[user_id]
            
            # Learning curve: Improve with each attempt at this skill
            learning_factor = min(0.3, 0.05 * skill_attempts[skill])
            
            # Final probability of correct answer
            prob_correct = min(0.95, base_prob + learning_factor)
            
            # Generate correctness
            correct = 1 if np.random.random() < prob_correct else 0
            correctness.append(correct)
            
            # Increment attempt count
            skill_attempts[skill] += 1
        
        # Add to dataset
        for i in range(n_user_interactions):
            data_rows.append({
                'user_id': user_id,
                'skill_id': skills[i],
                'correct': correctness[i]
            })
    
    # Create DataFrame
    data = pd.DataFrame(data_rows)
    
    print(f"Synthetic data shape: {data.shape}")
    print(f"Sample data:\n{data.head()}")
    
    return data

# Function to create knowledge tracing sequences with multiple window sizes
def create_kt_sequences(data, window_sizes=[3, 5, 7]):
    """
    Create knowledge tracing sequences with multiple window sizes.
    
    Args:
        data: DataFrame with columns user_id, skill_id, correct
        window_sizes: List of window sizes to use
    
    Returns:
        Dictionary with sequences for each window size
    """
    sequences = {}
    
    for window_size in window_sizes:
        X_skills = []  # Input skill sequences
        X_correct = []  # Input correctness sequences
        y = []  # Target correctness
        
        # Process each user separately
        for user_id in data['user_id'].unique():
            user_data = data[data['user_id'] == user_id].copy()
            
            # Skip users with too few interactions
            if len(user_data) <= window_size:
                continue
                
            # Get skills and correctness
            skills = user_data['skill_id'].values
            correct = user_data['correct'].values
            
            # Create sliding window sequences
            for i in range(window_size, len(skills)):
                # Input: last window_size interactions
                skill_seq = skills[i-window_size:i]
                correct_seq = correct[i-window_size:i]
                
                # Target: current interaction
                target_correct = correct[i]
                
                # Store sequence
                X_skills.append(skill_seq)
                X_correct.append(correct_seq)
                y.append(target_correct)
        
        # Convert to numpy arrays
        X_skills = np.array(X_skills, dtype=np.int32)
        X_correct = np.array(X_correct, dtype=np.int32)
        y = np.array(y, dtype=np.int32)
        
        sequences[window_size] = (X_skills, X_correct, y)
        print(f"Created {len(y)} examples with window size {window_size}")
    
    return sequences

# Build an enhanced knowledge tracing model with better architecture
def build_enhanced_kt_model(num_skills, window_size=5, embedding_dim=48):
    """
    Build an enhanced knowledge tracing model with improved architecture.
    
    Args:
        num_skills: Number of unique skills
        window_size: Size of input sequences
        embedding_dim: Dimension of embeddings
    
    Returns:
        Compiled model
    """
    # Input layers
    skill_input = Input(shape=(window_size,), dtype='int32', name='skill_input')
    correct_input = Input(shape=(window_size,), dtype='int32', name='correct_input')
    
    # Embedding layers with more dimensions
    skill_embedding = Embedding(
        input_dim=num_skills + 1,  # +1 for padding
        output_dim=embedding_dim,
        name='skill_embedding'
    )(skill_input)
    
    correct_embedding = Embedding(
        input_dim=2,  # 0 or 1
        output_dim=embedding_dim,
        name='correct_embedding'
    )(correct_input)
    
    # Try both approaches: LSTM and Flatten
    
    # 1. LSTM approach (captures sequential patterns)
    skill_lstm = LSTM(64, return_sequences=False)(skill_embedding)
    correct_lstm = LSTM(64, return_sequences=False)(correct_embedding)
    lstm_concat = Concatenate()([skill_lstm, correct_lstm])
    
    # 2. Flatten approach (simpler)
    skill_flat = Flatten()(skill_embedding)
    correct_flat = Flatten()(correct_embedding)
    flat_concat = Concatenate()([skill_flat, correct_flat])
    
    # Combine both approaches
    combined = Concatenate()([lstm_concat, flat_concat])
    
    # Deeper network with BatchNormalization
    x = Dense(256, activation='relu')(combined)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # Output layer
    output = Dense(1, activation='sigmoid')(x)
    
    # Create and compile model
    model = Model(inputs=[skill_input, correct_input], outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Function to train model with proper callbacks and learning rate schedule
def train_model(model, X_skills, X_correct, y, val_data, batch_size=128, epochs=20):
    """
    Train model with proper callbacks.
    
    Args:
        model: Compiled model
        X_skills, X_correct, y: Training data
        val_data: Validation data (X_skills_val, X_correct_val, y_val)
        batch_size: Batch size
        epochs: Maximum number of epochs
    
    Returns:
        Trained model and history
    """
    X_skills_val, X_correct_val, y_val = val_data
    
    # Custom callback to calculate AUC
    class AUCCallback(tf.keras.callbacks.Callback):
        def __init__(self, validation_data):
            super(AUCCallback, self).__init__()
            self.validation_data = validation_data
            self.val_auc = []
        
        def on_epoch_end(self, epoch, logs=None):
            X_skills_val, X_correct_val, y_val = self.validation_data
            y_pred = self.model.predict([X_skills_val, X_correct_val], verbose=0)
            val_auc = roc_auc_score(y_val, y_pred)
            logs['val_auc'] = val_auc
            self.val_auc.append(val_auc)
            print(f"Epoch {epoch+1} - val_auc: {val_auc:.4f}")
    
    auc_callback = AUCCallback((X_skills_val, X_correct_val, y_val))
    
    # Create early stopping callback that monitors val_auc
    early_stopping = EarlyStopping(
        monitor='val_auc',
        mode='max',
        patience=3,
        verbose=1,
        restore_best_weights=True
    )
    
    # Train model
    history = model.fit(
        [X_skills, X_correct],
        y,
        validation_data=([X_skills_val, X_correct_val], y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, auc_callback],
        verbose=1
    )
    
    # Add AUC to history
    history.history['val_auc'] = auc_callback.val_auc
    
    return model, history

# Function to build a simple knowledge tracing model (baseline)
def build_kt_model(num_skills, window_size=5, embedding_dim=32):
    """
    Build a simple knowledge tracing model.
    
    Args:
        num_skills: Number of unique skills
        window_size: Size of input sequences
        embedding_dim: Dimension of embeddings
    
    Returns:
        Compiled model
    """
    # Input layers
    skill_input = Input(shape=(window_size,), dtype='int32', name='skill_input')
    correct_input = Input(shape=(window_size,), dtype='int32', name='correct_input')
    
    # Embedding layers
    skill_embedding = Embedding(
        input_dim=num_skills + 1,  # +1 for padding
        output_dim=embedding_dim,
        name='skill_embedding'
    )(skill_input)
    
    correct_embedding = Embedding(
        input_dim=2,  # 0 or 1
        output_dim=embedding_dim,
        name='correct_embedding'
    )(correct_input)
    
    # Flatten embeddings
    skill_flat = Flatten()(skill_embedding)
    correct_flat = Flatten()(correct_embedding)
    
    # Concatenate embeddings
    concat = Concatenate()([skill_flat, correct_flat])
    
    # Hidden layers
    x = Dense(128, activation='relu')(concat)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    # Output layer
    output = Dense(1, activation='sigmoid')(x)
    
    # Create and compile model
    model = Model(inputs=[skill_input, correct_input], outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Main function
def main():
    """
    Main function.
    """
    print("Starting Knowledge Tracing Implementation with Repository Data")
    
    # Load datasets from repository
    print("Loading datasets from repository...")
    datasets = load_dkt_datasets()
    
    # The assistments dataset had parsing issues, but we can use the synthetic datasets
    if 'synthetic' in datasets and datasets['synthetic']:
        # Get the first synthetic dataset
        data_name = list(datasets['synthetic'].keys())[0]
        data = datasets['synthetic'][data_name]
        print(f"Using dataset: synthetic/{data_name}")
        
        # Convert the synthetic data to our required format
        # The synthetic data appears to be in a different format where each row is a student
        # and columns are questions/skills
        
        processed_data = []
        
        # Assuming the first column is student ID
        for idx, row in data.iterrows():
            user_id = idx + 1  # Use row index as user_id
            
            # Convert row to a sequence of interactions
            for col_idx, value in enumerate(row):
                if not pd.isna(value):  # Skip NaN values
                    skill_id = col_idx + 1  # Use column index as skill_id
                    correct = int(value)   # The value (0 or 1) indicates correctness
                    
                    processed_data.append({
                        'user_id': user_id,
                        'skill_id': skill_id,
                        'correct': correct
                    })
        
        # Convert to DataFrame
        processed_df = pd.DataFrame(processed_data)
        print(f"Processed data shape: {processed_df.shape}")
        
        # Create sequences
        window_size = 5
        X_skills, X_correct, y, num_skills = prepare_kt_sequences(processed_df, window_size=window_size)
        
        # Continue with training using the prepared sequences
        # Split data
        X_skills_train, X_skills_test, X_correct_train, X_correct_test, y_train, y_test = train_test_split(
            X_skills, X_correct, y, test_size=0.2, random_state=42
        )
        
        X_skills_train, X_skills_val, X_correct_train, X_correct_val, y_train, y_val = train_test_split(
            X_skills_train, X_correct_train, y_train, test_size=0.2, random_state=42
        )
        
        print(f"Train set: {len(y_train)} examples")
        print(f"Validation set: {len(y_val)} examples")
        print(f"Test set: {len(y_test)} examples")
        
        # Build and train models as in the original implementation
        # ...
    else:
        print("No datasets available. Falling back to synthetic data generation.")
        # Use the original synthetic data generation code
        data = generate_synthetic_data(n_users=300, n_skills=100)
        num_skills = data['skill_id'].max()
        
        # Create sequences
        all_sequences = create_kt_sequences(data, window_sizes=[window_size])
        X_skills, X_correct, y = all_sequences[window_size]
