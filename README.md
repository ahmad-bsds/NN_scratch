# Deep Neural Network from Scratch using NumPy

This project implements a simple Deep Neural Network (DNN) from scratch using only NumPy. It showcases fundamental deep learning concepts such as forward propagation, backward propagation, weight initialization, and gradient descent optimization. The network is trained on a synthetic dataset to perform binary classification.

## Features
- Custom implementation of a fully connected neural network.
- Support for ReLU and Sigmoid activation functions.
- Mean Squared Error (MSE) loss function.
- Gradient-based optimization using backpropagation.
- Configurable network architecture and hyperparameters.

## Motivation
This project is designed to demonstrate a deep understanding of the underlying principles of neural networks without relying on high-level libraries like TensorFlow or PyTorch. It is an excellent portfolio project for showcasing low-level implementation skills.

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Code Explanation](#code-explanation)
4. [Results](#results)
5. [Future Enhancements](#future-enhancements)
6. [Contributing](#contributing)

## Installation
Clone the repository and ensure you have Python installed.

```bash
git clone https://github.com/your-username/deep-learning-numpy.git
cd deep-learning-numpy
```

Install required libraries (NumPy):

```bash
pip install numpy
```

## Usage
1. Open the project in your favorite IDE or editor.
2. Run the Python script to train the model and observe results:

```bash
python deep_learning_numpy.py
```

## Code Explanation
### Neural Network Class
- **Initialization**: The network initializes weights and biases for all layers.
- **Forward Propagation**: Data flows through the network, applying activations and generating predictions.
- **Backward Propagation**: Gradients are computed layer by layer to update weights and biases.
- **Training Loop**: The network is trained for a specified number of epochs, minimizing the loss function.

### Example Usage
- **Dataset**: A synthetic dataset is generated where the goal is to classify whether the sum of two input features exceeds a threshold.
- **Training**: The network is trained on normalized input features with an adjustable learning rate and architecture.

## Results
The model achieves high accuracy on the synthetic dataset, learning the decision boundary effectively. Key metrics include:
- **Loss Reduction**: Observe the loss decreasing over epochs.
- **Accuracy**: Predictions achieve ~99% accuracy on the dataset.

## Future Enhancements
- Add support for additional activation functions like Tanh and Softmax.
- Implement other loss functions, such as Cross-Entropy.
- Extend the project to support multi-class classification problems.
- Explore advanced optimization algorithms like Adam or RMSprop.
- Train the network on real-world datasets.

## Contributing
Contributions are welcome! Feel free to fork the repository and submit pull requests.

1. Fork the project.
2. Create your feature branch:
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. Commit your changes:
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. Push to the branch:
   ```bash
   git push origin feature/AmazingFeature
   ```
5. Open a pull request.


---

Happy coding!

