import os
import numpy as np
import pandas as pd
import requests
import zipfile
import io

def load_dkt_datasets(repo_url="https://github.com/chrispiech/DeepKnowledgeTracing/archive/master.zip", 
                      local_dir="data", 
                      force_download=False):
    """
    Load datasets from the DeepKnowledgeTracing GitHub repository.
    
    Args:
        repo_url: URL to the repository zip file
        local_dir: Directory to store downloaded data
        force_download: Whether to force re-download even if files exist
        
    Returns:
        Dictionary with dataset names as keys and processed dataframes as values
    """
    # Create data directory if it doesn't exist
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    
    repo_dir = os.path.join(local_dir, "DeepKnowledgeTracing-master")
    
    # Download and extract repository if needed
    if force_download or not os.path.exists(repo_dir):
        print(f"Downloading repository from {repo_url}...")
        response = requests.get(repo_url)
        
        if response.status_code != 200:
            raise Exception(f"Failed to download from {repo_url}")
            
        print("Extracting files...")
        z = zipfile.ZipFile(io.BytesIO(response.content))
        z.extractall(local_dir)
    
    # Find all available datasets
    datasets = {}
    dataset_path = os.path.join(repo_dir, "data")
    
    if not os.path.exists(dataset_path):
        raise Exception(f"Dataset path {dataset_path} not found")
    
    # Process each dataset folder
    for dataset_folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, dataset_folder)
        
        if os.path.isdir(folder_path):
            print(f"Processing {dataset_folder} dataset...")
            datasets[dataset_folder] = {}
            
            # Process files in this dataset
            for file in os.listdir(folder_path):
                if file.endswith('.csv'):
                    file_path = os.path.join(folder_path, file)
                    print(f"  - Reading file: {file}")
                    
                    # Read the first few lines to understand the structure
                    with open(file_path, 'r', encoding='latin-1') as f:
                        first_lines = [f.readline().strip() for _ in range(5)]
                        
                    print(f"  - First lines of {file}:")
                    for i, line in enumerate(first_lines):
                        print(f"    Line {i+1}: {line}")
                    
                    # Try different parsing approaches
                    try:
                        # Custom parsing for assistments dataset which has special format
                        if dataset_folder == 'assistments':
                            # Parse by manually building records from the file
                            records = []
                            current_user = None
                            skills = []
                            corrects = []
                            
                            with open(file_path, 'r', encoding='latin-1') as f:
                                for line in f:
                                    line = line.strip()
                                    if not line:
                                        continue
                                        
                                    # Check if this is a user ID line (single number)
                                    if line.isdigit() and ',' not in line:
                                        # Save previous user if exists
                                        if current_user is not None and skills:
                                            for i, (skill, correct) in enumerate(zip(skills, corrects)):
                                                records.append({
                                                    'user_id': current_user,
                                                    'skill_id': skill,
                                                    'correct': correct,
                                                    'order': i
                                                })
                                        
                                        # Start new user
                                        current_user = int(line)
                                        skills = []
                                        corrects = []
                                    
                                    # Check if this is a skills line (comma-separated values)
                                    elif ',' in line and not line.startswith('skill'):
                                        parts = line.split(',')
                                        # Remove empty values
                                        parts = [p for p in parts if p.strip()]
                                        
                                        if all(p.isdigit() for p in parts):
                                            skills.extend(parts)
                                    
                                    # Check if this is a corrects line (comma-separated 0/1)
                                    elif ',' in line and all(c in '01,' for c in line):
                                        parts = line.split(',')
                                        # Remove empty values
                                        parts = [p for p in parts if p.strip()]
                                        
                                        if all(p in ['0', '1'] for p in parts):
                                            corrects.extend([int(p) for p in parts])
                            
                            # Save the last user
                            if current_user is not None and skills:
                                for i, (skill, correct) in enumerate(zip(skills, corrects)):
                                    records.append({
                                        'user_id': current_user,
                                        'skill_id': skill,
                                        'correct': correct,
                                        'order': i
                                    })
                            
                            data = pd.DataFrame(records)
                            print(f"  - Parsed {len(data)} records manually")
                        
                        # For other datasets, try standard CSV parsing
                        else:
                            # Try different delimiters and header configurations
                            for delimiter in [',', '\t', ';']:
                                try:
                                    data = pd.read_csv(file_path, delimiter=delimiter, encoding='latin-1')
                                    if len(data.columns) >= 3:  # At least user_id, skill_id, correct
                                        break
                                except Exception as e:
                                    continue
                            
                            # If standard parsing failed, try more manual approaches
                            if 'data' not in locals() or len(data.columns) < 3:
                                # Try to detect columns from header
                                with open(file_path, 'r', encoding='latin-1') as f:
                                    header_line = f.readline().strip()
                                    
                                if ',' in header_line:
                                    columns = [c.strip() for c in header_line.split(',')]
                                    data = pd.read_csv(file_path, names=columns, skiprows=1, 
                                                      delimiter=',', encoding='latin-1')
                        
                        # If we have data, standardize column names
                        if 'data' in locals() and not data.empty:
                            # Map columns to standard names
                            column_mappings = {}
                            
                            # Find user_id column
                            user_columns = [col for col in data.columns if 'user' in col.lower()]
                            if user_columns:
                                column_mappings[user_columns[0]] = 'user_id'
                            elif 'student' in data.columns:
                                column_mappings['student'] = 'user_id'
                            elif data.columns[0] not in ['user_id', 'skill_id', 'correct']:
                                column_mappings[data.columns[0]] = 'user_id'
                            
                            # Find skill_id column
                            skill_columns = [col for col in data.columns if 'skill' in col.lower() or 'problem' in col.lower()]
                            if skill_columns:
                                column_mappings[skill_columns[0]] = 'skill_id'
                            elif 'item' in data.columns:
                                column_mappings['item'] = 'skill_id'
                            elif len(data.columns) > 1 and data.columns[1] not in ['user_id', 'skill_id', 'correct']:
                                column_mappings[data.columns[1]] = 'skill_id'
                            
                            # Find correct column
                            correct_columns = [col for col in data.columns if 'correct' in col.lower() or 'score' in col.lower()]
                            if correct_columns:
                                column_mappings[correct_columns[0]] = 'correct'
                            elif 'answer' in data.columns:
                                column_mappings['answer'] = 'correct'
                            elif len(data.columns) > 2 and data.columns[2] not in ['user_id', 'skill_id', 'correct']:
                                column_mappings[data.columns[2]] = 'correct'
                            
                            # Rename columns
                            data = data.rename(columns=column_mappings)
                            
                            # Ensure we have the required columns
                            if not all(col in data.columns for col in ['user_id', 'skill_id', 'correct']):
                                print(f"  - Warning: Could not identify all required columns in {file}")
                                print(f"  - Available columns: {data.columns.tolist()}")
                                data = None
                            else:
                                # Convert to correct types
                                data['user_id'] = data['user_id'].astype(str)
                                data['skill_id'] = data['skill_id'].astype(str)
                                
                                # Try to convert 'correct' to binary
                                try:
                                    if data['correct'].dtype != int:
                                        # Handle different formats: True/False, "correct"/"incorrect", 1.0/0.0
                                        if data['correct'].dtype == bool:
                                            data['correct'] = data['correct'].astype(int)
                                        elif data['correct'].dtype == object:
                                            # If string values, map common patterns
                                            correct_map = {
                                                'true': 1, 'correct': 1, 'yes': 1, '1': 1, 'right': 1,
                                                'false': 0, 'incorrect': 0, 'no': 0, '0': 0, 'wrong': 0
                                            }
                                            data['correct'] = data['correct'].str.lower().map(
                                                lambda x: correct_map.get(x, 1 if x and x.strip() else 0)
                                            )
                                        else:
                                            # For numeric non-int types
                                            data['correct'] = (data['correct'] > 0).astype(int)
                                except Exception as e:
                                    print(f"  - Error converting 'correct' column: {e}")
                                    # Use a simpler approach
                                    data['correct'] = (data['correct'] > 0).astype(int)
                                
                                print(f"  - Successfully processed {file} with {len(data)} rows")
                    
                    except Exception as e:
                        print(f"  - Failed to process {file}: {e}")
                        data = None
                    
                    # Store the dataset if successful
                    if 'data' in locals() and data is not None and not data.empty:
                        datasets[dataset_folder][file.replace('.csv', '')] = data
                    else:
                        print(f"  - Skipping {file} due to parsing issues")
    
    # If no datasets were successfully loaded, raise exception
    if not any(datasets.values()):
        raise Exception("No datasets could be loaded successfully")
        
    print("\nLoaded datasets:")
    for dataset, files in datasets.items():
        print(f"- {dataset}: {list(files.keys())}")
    
    return datasets

def prepare_kt_sequences(data, window_size=5):
    """
    Prepare knowledge tracing sequences from a dataframe.
    
    Args:
        data: DataFrame with user_id, skill_id, correct columns
        window_size: Size of sliding window
        
    Returns:
        X_skills, X_correct, y arrays for model training
    """
    # Sort by user_id (and order if available)
    if 'order' in data.columns:
        data = data.sort_values(['user_id', 'order'])
    else:
        data = data.sort_values('user_id')
    
    # Convert skill_id to numeric indices
    skills = data['skill_id'].unique()
    skill_to_idx = {skill: idx+1 for idx, skill in enumerate(skills)}
    
    X_skills = []
    X_correct = []
    y = []
    
    # Process each user
    for user_id in data['user_id'].unique():
        user_data = data[data['user_id'] == user_id]
        
        # Skip users with too few interactions
        if len(user_data) <= window_size:
            continue
            
        # Get skills and correctness
        skills_seq = [skill_to_idx[skill] for skill in user_data['skill_id']]
        correct_seq = user_data['correct'].values
        
        # Create sequences
        for i in range(window_size, len(skills_seq)):
            X_skills.append(skills_seq[i-window_size:i])
            X_correct.append(correct_seq[i-window_size:i])
            y.append(correct_seq[i])
    
    # Convert to numpy arrays
    X_skills = np.array(X_skills, dtype=np.int32)
    X_correct = np.array(X_correct, dtype=np.int32)
    y = np.array(y, dtype=np.int32)
    
    print(f"Created {len(y)} sequences with window size {window_size}")
    print(f"Number of unique skills: {len(skill_to_idx)}")
    
    return X_skills, X_correct, y, len(skill_to_idx) + 1  # +1 for padding

# Example usage:
if __name__ == "__main__":
    # Load datasets
    datasets = load_dkt_datasets()
    
    # Choose one dataset for demonstration (e.g., assistments)
    if 'assistments' in datasets and datasets['assistments']:
        data_name = list(datasets['assistments'].keys())[0]
        data = datasets['assistments'][data_name]
        print(f"\nUsing dataset: assistments/{data_name}")
        
        # Prepare sequences
        X_skills, X_correct, y, num_skills = prepare_kt_sequences(data, window_size=5)
        
        print(f"\nSequence stats:")
        print(f"X_skills shape: {X_skills.shape}")
        print(f"X_correct shape: {X_correct.shape}")
        print(f"y shape: {y.shape}")
        print(f"Number of skills: {num_skills}")
    else:
        print("No assistments dataset available")
