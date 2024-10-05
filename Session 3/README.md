# Fully Connected (Linear) Layer

A **Fully Connected Layer** (also known as a **Linear Layer**) is a fundamental component in neural networks. In this layer, each neuron receives input from all neurons of the previous layer, which allows the model to learn complex relationships in the data.

## Mathematical Representation

For a fully connected layer, the output can be computed using the following equation:

\[
\mathbf{y} = \mathbf{W} \cdot \mathbf{x} + \mathbf{b}
\]

Where:

- \(\mathbf{y}\) is the output vector.
- \(\mathbf{W}\) is the weight matrix.
- \(\mathbf{x}\) is the input vector.
- \(\mathbf{b}\) is the bias vector.

### Matrix Shapes

- Let the input vector \(\mathbf{x}\) be of shape \( (n, 1) \), where \( n \) is the number of input features.
- The weight matrix \(\mathbf{W}\) has a shape of \( (m, n) \), where \( m \) is the number of neurons in the layer.
- The bias vector \(\mathbf{b}\) has a shape of \( (m, 1) \).
- The output vector \(\mathbf{y}\) will have a shape of \( (m, 1) \).

### Example

Consider an example where we have:

- An input vector \(\mathbf{x}\) with shape \( (3, 1) \):

\[
\mathbf{x} = \begin{pmatrix}
x_1 \\
x_2 \\
x_3
\end{pmatrix}
\]

- A weight matrix \(\mathbf{W}\) with shape \( (2, 3) \):

\[
\mathbf{W} = \begin{pmatrix}
w*{11} & w*{12} & w*{13} \\
w*{21} & w*{22} & w*{23}
\end{pmatrix}
\]

- A bias vector \(\mathbf{b}\) with shape \( (2, 1) \):

\[
\mathbf{b} = \begin{pmatrix}
b_1 \\
b_2
\end{pmatrix}
\]

The output \(\mathbf{y}\) will be calculated as follows:

\[
\mathbf{y} = \mathbf{W} \cdot \mathbf{x} + \mathbf{b}
\]

Performing the matrix multiplication and addition gives:

\[
\mathbf{y} = \begin{pmatrix}
w*{11}x_1 + w*{12}x*2 + w*{13}x*3 + b_1 \\
w*{21}x*1 + w*{22}x*2 + w*{23}x_3 + b_2
\end{pmatrix}
\]

## Activation Functions

Activation functions introduce non-linearity into the model, enabling it to learn complex patterns. Here are some important activation functions:

### 1. Sigmoid

The **Sigmoid** function is defined as:

\[
\sigma(x) = \frac{1}{1 + e^{-x}}
\]

### 2. ReLU (Rectified Linear Unit)

The **ReLU** function is defined as:

\[
\text{ReLU}(x) = \max(0, x)
\]

### 3. Tanh (Hyperbolic Tangent)

The **Tanh** function is defined as:

\[
\tanh(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}
\]

## Loss Functions

Loss functions measure the difference between the predicted output and the actual output. Here are some commonly used loss functions:

### 1. Mean Squared Error (MSE)

For regression tasks, the **Mean Squared Error** is defined as:

\[
\text{MSE} = \frac{1}{n} \sum\_{i=1}^{n} (y_i - \hat{y}\_i)^2
\]

Where:

- \(y_i\) is the true value.
- \(\hat{y}\_i\) is the predicted value.
- \(n\) is the number of observations.

### 2. Binary Cross-Entropy

For binary classification, the **Binary Cross-Entropy** loss is defined as:

\[
\text{BCE} = -\frac{1}{n} \sum\_{i=1}^{n} [y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)]
\]

### 3. Categorical Cross-Entropy

For multi-class classification, the **Categorical Cross-Entropy** loss is defined as:

\[
\text{CCE} = -\sum\_{i=1}^{C} y_i \log(\hat{y}\_i)
\]

Where:

- \(C\) is the number of classes.
- \(y_i\) is the true probability distribution (one-hot encoded).
- \(\hat{y}\_i\) is the predicted probability distribution.
