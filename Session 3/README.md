# README.md

## Neural Networks and Fully Connected Layers

### Introduction

Neural networks are powerful computational models inspired by the human brain. They consist of layers of interconnected nodes (or neurons) that learn to perform tasks by adjusting the connections (weights) between these neurons. Neural networks are widely used in applications like classification, regression, image recognition, natural language processing, and more.

One of the key components of a neural network is the **Fully Connected Layer** (also known as a Dense Layer). In a fully connected layer, every neuron is connected to all neurons in the previous layer, enabling the model to learn from all available features and create complex patterns from the input data.

---

### Fully Connected Layer: Mathematical Representation

A fully connected layer performs a linear transformation on the input followed by the application of a non-linear activation function. The operation in a fully connected layer can be represented as:

\[
Y = f(WX + b)
\]

Where:

- \(X\): Input vector (of size \(n\), representing the number of features).
- \(W\): Weight matrix (of size \(m \times n\), where \(m\) is the number of neurons in the layer and \(n\) is the number of input features).
- \(b\): Bias vector (of size \(m\), which is added to each neuron's output).
- \(f\): Activation function (e.g., ReLU, Sigmoid, Tanh).
- \(Y\): Output vector (of size \(m\), representing the output of the layer.

#### Matrix Representation of the Fully Connected Layer

The fully connected layer is often represented as a matrix operation:

\[
Y = f(WX + b)
\]

Where:

- \(X\) is the input vector of size \((n, 1)\),
- \(W\) is the weight matrix of size \((m, n)\),
- \(b\) is the bias vector of size \((m, 1)\),
- \(Y\) is the output vector of size \((m, 1)\).

---

### Batched Inputs and Matrix Dimensions

In many real-world scenarios, it is common to process multiple inputs at once, known as **batched inputs**. When dealing with batched inputs, the input \(X\) becomes a matrix rather than a vector. The dimensions for batched inputs are as follows:

- **Batch Size (B)**: The number of input examples processed at the same time.
- **Input Matrix (X)**: \( (B, n) \), where \( B \) is the batch size and \( n \) is the number of features for each input.
- **Weight Matrix (W)**: \( (m, n) \), where \( m \) is the number of neurons and \( n \) is the number of input features.
- **Bias Vector (b)**: \( (m, 1) \), bias for each neuron.
- **Output Matrix (Y)**: \( (B, m) \), where each row represents the output for one input example.

---

### Matrix Multiplication for Unbatched Inputs

For a single input (unbatched), the operation in a fully connected layer can be represented as:

\[
Y = W \cdot X + b
\]

Where:

- \(X\) is the input vector of size \((n, 1)\),
- \(W\) is the weight matrix of size \((m, n)\),
- \(b\) is the bias vector of size \((m, 1)\),
- \(Y\) is the output vector of size \((m, 1)\).

#### Example (Unbatched Input):

- Let the input \( X \) be of size \( (3, 1) \) (3 features),
- Let the weight matrix \( W \) be of size \( (4, 3) \) (4 neurons, 3 input features),
- Let the bias vector \( b \) be of size \( (4, 1) \).

The resulting output \( Y \) will be of size \( (4, 1) \).

---

### Matrix Multiplication for Batched Inputs

When processing multiple inputs at the same time (batched inputs), the matrix multiplication is generalized as:

\[
Z = X \cdot W^T + b
\]

Where:

- \(X\) is the input matrix of size \((B, n)\),
- \(W^T\) is the transposed weight matrix of size \((n, m)\),
- \(b\) is the bias vector of size \((m, 1)\), broadcasted across all batch inputs,
- \(Z\) is the output matrix of size \((B, m)\), where each row corresponds to the output for one input example.

#### Example (Batched Inputs):

- Let the batch size \( B = 5 \) (i.e., 5 input examples),
- Let the input matrix \( X \) be of size \( (5, 3) \) (5 examples, each with 3 features),
- Let the weight matrix \( W \) be of size \( (4, 3) \) (4 neurons, 3 input features),
- Let the bias vector \( b \) be of size \( (4, 1) \).

The resulting output matrix \( Y \) will be of size \( (5, 4) \).

---

### Activation Functions

After performing the linear transformation, an activation function \( f \) is applied element-wise to the output. Some common activation functions include:

- **ReLU (Rectified Linear Unit)**:

  f(x) = \text{max}(0, x)

  ReLU is widely used due to its ability to mitigate the vanishing gradient problem.

- **Sigmoid**:

  f(x) = \frac{1}{1 + e^{-x}}

  Sigmoid is useful for binary classification tasks, as it squashes the input between 0 and 1.

- **Tanh (Hyperbolic Tangent)**:

  f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}

  Tanh outputs values between -1 and 1, making it suitable for tasks where outputs should be centered around zero.

---

### Conclusion

The **fully connected layer** is one of the core components of a neural network, transforming input data through linear operations and applying activation functions to model complex relationships. With batched inputs, neural networks can efficiently process multiple examples in parallel, using matrix operations for performance optimization.

Understanding the mathematical formulation of fully connected layers is crucial for designing and implementing efficient neural networks for a wide range of applications.
