import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, Flatten, Concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from load_dkt_data import load_dkt_datasets

# Disable TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def build_kt_model(num_skills, window_size=5, embedding_dim=32):
    """Simple knowledge tracing model"""
    # Input layers
    skill_input = Input(shape=(window_size,), dtype='int32', name='skill_input')
    correct_input = Input(shape=(window_size,), dtype='int32', name='correct_input')
    
    # Embedding layers
    skill_embedding = Embedding(input_dim=num_skills + 1, output_dim=embedding_dim)(skill_input)
    correct_embedding = Embedding(input_dim=2, output_dim=embedding_dim)(correct_input)
    
    # Flatten embeddings
    skill_flat = Flatten()(skill_embedding)
    correct_flat = Flatten()(correct_embedding)
    
    # Concatenate
    concat = Concatenate()([skill_flat, correct_flat])
    
    # Dense layers
    x = Dense(128, activation='relu')(concat)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    # Output layer
    output = Dense(1, activation='sigmoid')(x)
    
    # Create model
    model = Model(inputs=[skill_input, correct_input], outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def process_repo_data(data):
    """Process repository data into the format needed for KT"""
    processed_data = []
    
    # Each row is a student, columns are skills/questions
    for idx, row in data.iterrows():
        user_id = idx + 1  # Use row index as user_id
        
        # Convert row to a sequence of interactions
        for col_idx, value in enumerate(row):
            if pd.notna(value):  # Skip NaN values
                skill_id = col_idx + 1  # Use column index as skill_id
                correct = int(float(value))   # Convert to int
                
                processed_data.append({
                    'user_id': user_id,
                    'skill_id': skill_id,
                    'correct': correct
                })
    
    # Convert to DataFrame
    return pd.DataFrame(processed_data)

def prepare_kt_sequences(data, window_size=5):
    """Prepare knowledge tracing sequences"""
    X_skills = []
    X_correct = []
    y = []
    
    # Process each user
    for user_id in data['user_id'].unique():
        user_data = data[data['user_id'] == user_id].sort_values('skill_id')
        
        if len(user_data) <= window_size:
            continue
            
        skills = user_data['skill_id'].values
        corrects = user_data['correct'].values
        
        # Create sequences
        for i in range(window_size, len(skills)):
            X_skills.append(skills[i-window_size:i])
            X_correct.append(corrects[i-window_size:i])
            y.append(corrects[i])
    
    return np.array(X_skills), np.array(X_correct), np.array(y)

def main():
    """Main function"""
    print("=" * 50)
    print("KNOWLEDGE TRACING IMPLEMENTATION")
    print("=" * 50)
    
    # Load datasets
    print("Loading datasets from repository...")
    datasets = load_dkt_datasets()
    
    # Use synthetic dataset
    if 'synthetic' in datasets and datasets['synthetic']:
        data_name = list(datasets['synthetic'].keys())[0]
        data = datasets['synthetic'][data_name]
        print(f"Using synthetic dataset: {data_name}")
        
        # Process data
        processed_data = process_repo_data(data)
        print(f"Processed data shape: {processed_data.shape}")
        
        # Get number of skills
        num_skills = processed_data['skill_id'].max()
        print(f"Number of skills: {num_skills}")
        
        # Prepare sequences
        window_size = 5
        X_skills, X_correct, y = prepare_kt_sequences(processed_data, window_size)
        print(f"Created {len(y)} sequences with window size {window_size}")
        
        # Split data
        X_skills_train, X_skills_test, X_correct_train, X_correct_test, y_train, y_test = train_test_split(
            X_skills, X_correct, y, test_size=0.2, random_state=42
        )
        
        print(f"Train set: {len(y_train)} examples")
        print(f"Test set: {len(y_test)} examples")
        
        # Build model
        model = build_kt_model(num_skills, window_size)
        print("Model built successfully")
        
        # Train model (for just a few epochs)
        print("Training model...")
        model.fit(
            [X_skills_train, X_correct_train],
            y_train,
            validation_split=0.2,
            epochs=3,
            batch_size=64
        )
        
        # Evaluate model
        print("\nEvaluating model...")
        y_pred = model.predict([X_skills_test, X_correct_test])
        auc = roc_auc_score(y_test, y_pred)
        binary_pred = (y_pred >= 0.5).astype(int)
        accuracy = accuracy_score(y_test, binary_pred)
        
        # Create table similar to Table 1 in the paper
        results_table = pd.DataFrame({
            'Dataset': ['Synthetic (Repository)'],
            'Model': ['KT-Simple'],
            'AUC': [auc],
            'Accuracy': [accuracy]
        })
        
        print("\nResults (Similar to Table 1 in the paper):")
        print(results_table.to_string(index=False))
    else:
        print("No synthetic datasets available")
    
    print("\nImplementation completed")

# Make sure main is called when script is run
if __name__ == "__main__":
    main()
