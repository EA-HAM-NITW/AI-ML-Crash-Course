# Fully Connected Layer in Neural Networks

## Introduction

A **Fully Connected Layer** (also known as a Linear Layer or Dense Layer) is a fundamental building block in neural networks. In this layer, each neuron is connected to every neuron in the previous layer, allowing the model to learn from all available features. Fully connected layers are typically used at the end of convolutional neural networks or as part of multi-layer perceptrons (MLPs).

---

## Mathematical Representation

The operation of a fully connected layer can be mathematically represented as follows:

\[
Y = f(WX + b)
\]

Where:

- \(Y\): Output vector.
- \(W\): Weight matrix.
- \(X\): Input vector.
- \(b\): Bias vector.
- \(f\): Activation function applied element-wise.

### Matrix Representation

For a fully connected layer, the relationship can be expressed in matrix form. If we have:

- \(X\) as the input vector of size \(n\) (number of features),
- \(W\) as the weight matrix of size \(m \times n\) (where \(m\) is the number of neurons),
- \(b\) as the bias vector of size \(m\).

The output \(Y\) will be of size \(m\) and can be computed as:

\[
Y = W \cdot X + b
\]

### Example with Matrices

Let's consider a simple example with matrices.

#### Input

Assume:

- \(X\) is an input vector:

\[
X = \begin{bmatrix} 0.5 \\ 1.0 \\ -1.5 \end{bmatrix} \quad \text{(Size: \(3 \times 1\))}
\]

#### Weight Matrix

Let \(W\) be the weight matrix:

\[
W = \begin{bmatrix} 0.2 & 0.8 & -0.5 \\ 1.0 & -1.0 & 0.5 \end{bmatrix} \quad \text{(Size: \(2 \times 3\))}
\]

#### Bias Vector

Let \(b\) be the bias vector:

\[
b = \begin{bmatrix} 0.1 \\ -0.2 \end{bmatrix} \quad \text{(Size: \(2 \times 1\))}
\]

#### Output Calculation

Now we calculate the output \(Y\):

\[
Y = W \cdot X + b
\]

Calculating \(W \cdot X\):

\[
W \cdot X = \begin{bmatrix} 0.2 & 0.8 & -0.5 \\ 1.0 & -1.0 & 0.5 \end{bmatrix} \cdot \begin{bmatrix} 0.5 \\ 1.0 \\ -1.5 \end{bmatrix} = \begin{bmatrix} (0.2 \cdot 0.5) + (0.8 \cdot 1.0) + (-0.5 \cdot -1.5) \\ (1.0 \cdot 0.5) + (-1.0 \cdot 1.0) + (0.5 \cdot -1.5) \end{bmatrix}
\]

\[
= \begin{bmatrix} 0.1 + 0.8 + 0.75 \\ 0.5 - 1.0 - 0.75 \end{bmatrix} = \begin{bmatrix} 1.65 \\ -1.25 \end{bmatrix} \quad \text{(Size: \(2 \times 1\))}
\]

Now adding the bias \(b\):

\[
Y = \begin{bmatrix} 1.65 \\ -1.25 \end{bmatrix} + \begin{bmatrix} 0.1 \\ -0.2 \end{bmatrix} = \begin{bmatrix} 1.75 \\ -1.45 \end{bmatrix} \quad \text{(Size: \(2 \times 1\))}
\]

Thus, the output \(Y\) is:

\[
Y = \begin{bmatrix} 1.75 \\ -1.45 \end{bmatrix}
\]

---

## Activation Functions

Activation functions introduce non-linearity into the model, allowing it to learn complex relationships. Some common activation functions include:

### 1. ReLU (Rectified Linear Unit)

The ReLU function is defined as:

\[
f(x) = \text{max}(0, x)
\]

It is widely used due to its simplicity and ability to mitigate the vanishing gradient problem.

### 2. Sigmoid

The Sigmoid function squashes the input to a range between 0 and 1:

\[
f(x) = \frac{1}{1 + e^{-x}}
\]

This is useful for binary classification tasks.

### 3. Tanh (Hyperbolic Tangent)

The Tanh function outputs values between -1 and 1:

\[
f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
\]

It is often preferred over the sigmoid function due to its zero-centered outputs.

---

## Loss Functions

Loss functions measure the difference between the predicted output and the actual target values. Some important loss functions include:

### 1. Mean Squared Error (MSE)

MSE is commonly used for regression tasks:

\[
\text{MSE} = \frac{1}{N} \sum\_{i=1}^{N} (y_i - \hat{y}\_i)^2
\]

Where \(y_i\) is the actual value, \(\hat{y}\_i\) is the predicted value, and \(N\) is the number of observations.

### 2. Binary Cross-Entropy Loss

Used for binary classification tasks:

\[
\text{Loss} = -\frac{1}{N} \sum\_{i=1}^{N} [y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)]
\]

Where \(y_i\) is the true label (0 or 1) and \(\hat{y}\_i\) is the predicted probability.

### 3. Categorical Cross-Entropy Loss

Used for multi-class classification tasks:

\[
\text{Loss} = -\sum\_{i=1}^{C} y_i \log(\hat{y}\_i)
\]

Where \(C\) is the number of classes, \(y_i\) is the true distribution (one-hot encoded) and \(\hat{y}\_i\) is the predicted probability for each class.

---

## Conclusion

The fully connected layer is a critical component of neural networks, enabling them to model complex relationships through linear transformations followed by non-linear activation functions. Understanding the mathematical principles and the role of activation and loss functions is essential for designing effective neural networks for various applications.
