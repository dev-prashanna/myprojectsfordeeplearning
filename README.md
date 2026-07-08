# Deep Learning from Scratch

[![Python](https://img.shields.io/badge/Python-3.10+-yellow)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)](https://jupyter.org/)

Implementation of deep learning algorithms from scratch to understand the fundamentals.

## Overview

This repository contains implementations of core deep learning concepts built from scratch using only NumPy and basic Python. The goal is to understand how neural networks work internally, not just use frameworks.

## Project Structure

```
myprojectsfordeeplearning/
├── notebooks/
│   ├── gradient_descent_scratch.ipynb    # Gradient descent implementation
│   ├── intercept_calc_gd.ipynb           # Linear regression with GD
│   └── opencv_practise.ipynb             # OpenCV experiments
│
├── images/                               # Reference images
│   └── images.jpeg
│
├── .gitignore
├── LICENSE                               # MIT License
└── README.md                             # This file
```

## Notebooks

### 1. Gradient Descent from Scratch

**File:** `notebooks/gradient_descent_scratch.ipynb`

Implement gradient descent optimization algorithm from scratch:
- Cost function computation
- Gradient calculation
- Parameter updates
- Convergence visualization

### 2. Linear Regression with Gradient Descent

**File:** `notebooks/intercept_calc_gd.ipynb`

Build linear regression using gradient descent:
- Simple linear regression
- Multiple linear regression
- Feature scaling
- Learning rate tuning

### 3. OpenCV Practice

**File:** `notebooks/opencv_practise.ipynb`

Computer vision experiments with OpenCV:
- Image loading and display
- Color space transformations
- Edge detection
- Basic image processing

## Installation

### Prerequisites

- Python 3.10+
- Jupyter Notebook

### Setup

```bash
# Clone the repository
git clone https://github.com/dev-prashanna/myprojectsfordeeplearning.git
cd myprojectsfordeeplearning

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install dependencies
pip install numpy matplotlib jupyter opencv-python
```

## Usage

```bash
# Start Jupyter
jupyter notebook

# Navigate to notebooks/ folder
# Open any .ipynb file and run cells
```

## Key Concepts Implemented

### Gradient Descent

```python
# Basic gradient descent update rule
def gradient_descent(X, y, learning_rate=0.01, epochs=100):
    weights = np.zeros(X.shape[1])
    bias = 0
    
    for epoch in range(epochs):
        # Forward pass
        y_pred = X.dot(weights) + bias
        
        # Compute loss
        loss = np.mean((y_pred - y) ** 2)
        
        # Compute gradients
        dw = (2/X.shape[0]) * X.T.dot(y_pred - y)
        db = (2/X.shape[0]) * np.sum(y_pred - y)
        
        # Update parameters
        weights -= learning_rate * dw
        bias -= learning_rate * db
    
    return weights, bias
```

### Loss Functions

- **MSE (Mean Squared Error):** `L = (1/n) * sum((y_pred - y)^2)`
- **MAE (Mean Absolute Error):** `L = (1/n) * sum(|y_pred - y|)`

### Activation Functions

- **Sigmoid:** `σ(x) = 1 / (1 + e^(-x))`
- **ReLU:** `f(x) = max(0, x)`
- **Tanh:** `tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))`

## Learning Outcomes

From these notebooks, you will learn:

- How gradient descent optimizes parameters
- The math behind backpropagation
- Why feature scaling matters
- How learning rate affects convergence
- Basic computer vision with OpenCV

## Future Work

- [ ] Implement neural network from scratch
- [ ] Add backpropagation implementation
- [ ] Implement different optimizers (Adam, RMSprop)
- [ ] Add regularization techniques
- [ ] Create convolutional neural network
- [ ] Add recurrent neural network

## References

- [Deep Learning Book](https://www.deeplearningbook.org/)
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)
- [Stanford CS231n](http://cs231n.stanford.edu/)
- [3Blue1Brown Neural Networks](https://www.3blue1brown.com/topics/neural-networks)

## License

This project is licensed under the MIT License -- see the [LICENSE](LICENSE) file for details.

## Author

**Prashanna Tiwari**
- GitHub: [@dev-prashanna](https://github.com/dev-prashanna)
- LinkedIn: [Prashanna Tiwari](https://www.linkedin.com/in/prashanna-tiwari-1b9a01398/)
