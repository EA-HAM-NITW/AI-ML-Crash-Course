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

- \( X \): Input vector (of size \( n \), representing the number of features).
- \( W \): Weight matrix (of size \( m \times n \), where \( m \) is the number of neurons in the layer and \( n \) is the number of input features.
- \( b \): Bias vector (of size \( m \), which is added to each neuron's output.
- \( f \): Activation function (e.g., ReLU, Sigmoid, Tanh).
- \( Y \): Output vector (of size \( m \), representing the output of the layer.

#### Matrix Representation of the Fully Connected Layer

The fully connected layer is often represented as a matrix operation:

\[
Y = f(WX + b)
\]

Where:

- \( X \): Input vector of size \( (n, 1) \).
- \( W \): Weight matrix of size \( (m, n) \).
- \( b \): Bias vector of size \( (m, 1) \).
- \( Y \): Output vector of size \( (m, 1) \).

---

### Batched Inputs and Matrix Dimensions

In many real-world scenarios, it is common to process multiple inputs at once, known as **batched inputs**. When dealing with batched inputs, the input \( X \) becomes a matrix rather than a vector. The dimensions for batched inputs are as follows:

- **Batch Size (B)**: The number of input examples processed at the same time.
- **Input Matrix (X)**: \( (B, n) \), where \( B \) is the batch size and \( n \) is the number of features for each input.
- **Weight Matrix (W)**: \( (m, n) \), where \( m \) is the number of neurons and \( n \) is the number of input features.
- **Bias Vector (b)**: \( (m, 1) \), bias for each neuron.
- **Output Matrix (Y)**: \( (B, m) \), where each row represents the output for one input example.

The operation becomes:

\[
Z = XW^T + b
\]

Where:

- \( Z \) is the pre-activation output of size \( (B, m) \),
- \( W^T \) is the transpose of the weight matrix, allowing matrix multiplication with the input \( X \),
- \( b \) is the bias vector, which is broadcasted across all batch inputs.

#### Example of Matrix Sizes for Batched Inputs

Let’s consider an example where:

- **Batch Size (B)** = 4 (i.e., processing 4 input examples simultaneously),
- **Input Size (n)** = 3 (i.e., each input has 3 features),
- **Neurons in Fully Connected Layer (m)** = 5 (i.e., there are 5 neurons in the layer).

In this case:

- Input matrix \( X \): \( (4, 3) \),
- Weight matrix \( W \): \( (5, 3) \),
- Bias vector \( b \): \( (5, 1) \),
- Output matrix \( Y \): \( (4, 5) \).

---

### Activation Functions

After performing the linear transformation, an activation function \( f \) is applied element-wise to the output. Some common activation functions include:

- **ReLU (Rectified Linear Unit)**:
  \[
  f(x) = \text{max}(0, x)
  \]
  ReLU is widely used due to its ability to mitigate the vanishing gradient problem.

- **Sigmoid**:
  \[
  f(x) = \frac{1}{1 + e^{-x}}
  \]
  Sigmoid is useful for binary classification tasks, as it squashes the input between 0 and 1.

- **Tanh (Hyperbolic Tangent)**:
  \[
  f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
  \]
  Tanh outputs values between -1 and 1, making it suitable for tasks where outputs should be centered around zero.

---

### Conclusion

The **fully connected layer** is one of the core components of a neural network, transforming input data through linear operations and applying activation functions to model complex relationships. With batched inputs, neural networks can efficiently process multiple examples in parallel, using matrix operations for performance optimization.

Understanding the mathematical formulation of fully connected layers is crucial for designing and implementing efficient neural networks for a wide range of applications.
