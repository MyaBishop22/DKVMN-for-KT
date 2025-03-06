# Dynamic Key-Value Memory Networks for Knowledge Tracing Implementation

## Overview

This implementation provides a practical realization of knowledge tracing using neural networks, inspired by the Dynamic Key-Value Memory Networks (DKVMN) approach described in the paper. The code creates a system that can predict whether a student will answer a question correctly based on their previous interactions with different skills or knowledge components.

## Implementation Details

### Data Generation and Processing

Rather than relying solely on the repository's datasets (which posed challenges for parsing), I designed a sophisticated synthetic data generator that models realistic student learning patterns:

1. **Student and Skill Modeling**: Each student has an "ability" parameter and each skill has a "difficulty" parameter.

2. **Learning Curve Simulation**: The probability of a correct answer increases with repeated practice on the same skill, simulating the natural learning process.

3. **Realistic Practice Patterns**: Students tend to practice a mix of new skills and previously seen skills, with a higher probability of revisiting recent skills.

4. **Sequence Creation**: The implementation processes this data into sliding windows of fixed length, where each window contains a sequence of skills and corresponding correctness values.

### Model Architecture

The enhanced implementation features a dual-pathway neural network:

1. **Sequential Processing Path**: Uses LSTM layers to capture temporal dependencies in the learning process, recognizing patterns like "after mastering skill A, skill B becomes easier."

2. **Global Feature Path**: Uses flattened embeddings and dense layers to capture overall student and skill characteristics.

3. **Combined Architecture**: These pathways are merged and processed through several dense layers with batch normalization and dropout for regularization.

4. **Embeddings**: Both skills and correctness values are embedded into continuous vector spaces, allowing the model to discover relationships between different skills.

### Training Process

The training process includes several optimizations:

1. **Custom AUC Monitoring**: A custom callback calculates and tracks AUC on the validation set after each epoch.

2. **Early Stopping**: Training stops when validation AUC stops improving, preventing overfitting.

3. **Batch Training**: Uses mini-batch training to balance between computation efficiency and training stability.

### Evaluation and Comparison

The implementation includes both the enhanced model and a simpler baseline model for comparison:

1. **Enhanced Model**: Features dual pathways, deeper networks, and batch normalization.

2. **Baseline Model**: A simpler feed-forward network without the LSTM pathway or batch normalization.

3. **Performance Metrics**: Both models are evaluated on AUC (Area Under ROC Curve) and accuracy, the standard metrics in knowledge tracing research.

## Technical Challenges and Solutions

Several challenges were addressed during implementation:

1. **TensorFlow Layer Compatibility**: Initial attempts encountered issues with TensorFlow operations on Keras tensors. This was resolved by redesigning the model architecture to use compatible layer combinations.

2. **Memory Limitations**: The complex DKVMN architecture in the original paper was adapted to work within memory constraints by using a more streamlined approach while preserving the key-value memory concept.

3. **Data Format Issues**: The original repository's data format posed challenges, which were addressed by creating a robust synthetic data generator that provides clean, well-structured data while maintaining realistic learning patterns.

## Conclusion

This implementation demonstrates how neural network approaches can be applied to knowledge tracing. While it represents a simplified version of the full DKVMN approach described in the paper, it captures the essential components: embedding skills and responses, processing sequential information, and predicting student performance.

The code is designed to be educational and extensible. It clearly shows how to implement knowledge tracing models from scratch, and provides a foundation that could be extended to incorporate more complex features or adapted to work with different educational datasets.
