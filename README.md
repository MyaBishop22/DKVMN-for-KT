# Dynamic Key-Value Memory Networks for Knowledge Tracing Implementation

## Overview

This implementation provides a practical approach to knowledge tracing using neural networks inspired by the Dynamic Key-Value Memory Networks (DKVMN) paper. The code creates a system that can predict whether a student will answer a question correctly based on their history of interactions with different skills or knowledge components.

## Implementation Approach

### Data Processing

The implementation works with synthetic datasets from the DeepKnowledgeTracing repository, which represent student-skill interactions in a matrix format:

1. **Data Extraction**: The data loader downloads the repository and extracts the synthetic datasets, where each row represents a student and each column represents a skill/question.

2. **Data Transformation**: The implementation converts this matrix format into a sequence format where each row represents a single interaction (user_id, skill_id, correct), which is the standard format for knowledge tracing.

3. **Sequence Creation**: For each student, we create sliding windows of interactions to capture the sequential nature of learning. Each window consists of previous skill interactions and correctness values, with the next interaction serving as the prediction target.

### Model Architecture

The implementation uses a simplified but effective neural network architecture:

1. **Embedding Layers**: Separate embedding layers for skills and correctness values convert categorical identifiers into continuous vector representations, allowing the model to learn relationships between skills.

2. **Flattened Representation**: The embeddings are flattened to create a fixed-length representation of the student's knowledge state based on their interaction history.

3. **Dense Layers**: Multiple dense layers with dropout for regularization process this representation to make the prediction.

4. **Output Layer**: A sigmoid activation function produces a probability that the student will answer the next question correctly.

### Training and Evaluation

The model is trained and evaluated with a focus on standard knowledge tracing metrics:

1. **Training Process**: The model is trained on 80% of the data, with 20% of that set aside for validation to monitor for overfitting.

2. **Evaluation Metrics**: The model is evaluated on Area Under the ROC Curve (AUC) and accuracy, the standard metrics in the knowledge tracing literature and used in Table 1 of the paper.

3. **Results Presentation**: Results are presented in a table format similar to Table 1 in the paper for easy comparison.

## Running the Implementation

### Prerequisites

To run this implementation, you need:

1. **Python 3.9-3.11**: The code is compatible with these Python versions.

2. **Required Libraries**:
   - TensorFlow 2.x (for building and training the neural network)
   - NumPy (for numerical operations)
   - Pandas (for data manipulation)
   - scikit-learn (for evaluation metrics and data splitting)
   - requests (for downloading the repository)

### Installation

Install the required libraries:

```bash
pip install tensorflow numpy pandas scikit-learn requests
```

### Running the Code

1. **Save the Two Required Files**:
   - Save `kt_fixed.py` - The main implementation file
   - Save `load_dkt_data.py` - The data loading module
   
   Both files should be in the same directory.

2. **Execute the Implementation**:
   ```bash
   python kt_fixed.py
   ```

3. **What to Expect**:
   - The script will first download the DeepKnowledgeTracing repository (if not already downloaded)
   - It will process a synthetic dataset and prepare sequences
   - The model will be built and trained for 3 epochs
   - Finally, evaluation results will be displayed in a format similar to Table 1 in the paper

The entire process takes a few minutes, with most of the time spent downloading and processing the dataset, and training the model.

## Customization

You can modify the implementation in several ways:

- Change the `window_size` parameter to experiment with different history lengths
- Adjust the model architecture by modifying the `build_kt_model` function
- Change the number of training epochs for more refined results
- Try different synthetic datasets by modifying the dataset selection logic
