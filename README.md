# Neural Network Implementation with L1 and L2 Regularization

This repository contains an implementation of a fully connected multilayer feedforward neural network with L1 and L2 regularization, along with experiments on multiple regression datasets.

## Features

- Custom neural network implementation from scratch using NumPy
- Support for both L1 and L2 regularization
- Multiple activation functions (ReLU, Softmax)
- Configurable network architecture with variable hidden layers
- Support for different loss functions (MSE, Cross-entropy)
- Mini-batch gradient descent training
- Comprehensive visualization tools for analysis

## Implementation Details

### Neural Network Architecture
- Fully connected layers with configurable sizes
- ReLU activation for hidden layers
- Optional Softmax activation for output layer
- Support for both regression and classification tasks

### Regularization
- L1 regularization (Lasso)
- L2 regularization (Ridge)
- Configurable regularization strengths

### Training Features
- Mini-batch gradient descent
- Configurable learning rate
- Support for multiple loss functions
- Built-in training and validation split functionality

## Experiments

The implementation was tested on three different datasets:

1. Multi-target Regression (2 features, 3 targets)
2. Multi-linear Regression (10 features, 3 targets)
3. Univariate Regression (5 features, 1 target)

### Analysis Tools
- Training and validation loss curves
- Weight distribution visualization
- Feature importance analysis
- Error distribution plots
- Bias-variance analysis
- Actual vs predicted visualization in 3D

## Usage

```python
# Initialize the model
model = FullyConnectedNeuralNetwork(
    input_size=2,
    output_size=3,
    hidden_layers=[64, 64],
    loss_function='mse',
    learning_rate=0.01,
    l1_reg=0.01,
    l2_reg=0.0
)

# Train the model
model.forward_pass(X)
model.backpropagate(X, y, activations)
```

## Dependencies

- NumPy
- Matplotlib
- mpl_toolkits.mplot3d (for 3D visualizations)

## Results

The experiments demonstrate:
- Effects of different regularization techniques on model performance
- Impact of network architecture on learning
- Trade-offs between model complexity and generalization
- Comparison between L1 and L2 regularization effects

## Contributing

Feel free to open issues or submit pull requests with improvements.

## License

[MIT License](LICENSE)
