# Introduction to Logistic Regression

## 1. What is Logistic Regression?

Logistic regression is a **classification algorithm** that predicts the probability of a binary outcome (0 or 1). Despite its name, it is used for classification tasks, not regression. It estimates the probability that an instance belongs to a particular class using a **sigmoid function**.

---

## 2. Difference Between Linear and Logistic Regression

- **Linear Regression**: Used for predicting continuous values.
- **Logistic Regression**: Used for predicting probabilities and classifying into categories (usually binary).

---

## 3. Sigmoid Function

The sigmoid function transforms the linear equation into a probability value:
\[
\sigma(z) = rac{1}{1 + e^{-z}}
\]
The output is a probability between 0 and 1. The decision boundary is typically set at 0.5: if the output is greater than 0.5, the instance is classified as 1 (positive), otherwise, it is 0 (negative).

---

## 4. Logistic Regression Hypothesis Function

The logistic regression hypothesis is:
\[
h\_ heta(x) = \sigma( heta_0 + heta_1 x_1 + \dots + heta_n x_n)
\]
Where `theta` represents the learned parameters, and `x` represents the input features.

---

## 5. Cost Function (Log Loss)

The cost function for logistic regression, called **log loss** or **binary cross-entropy**, is:
\[
J( heta) = -rac{1}{m} \sum*{i=1}^{m} \left[ y^{(i)} \log(h* heta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h\_ heta(x^{(i)}))
ight]
\]
This function helps to minimize the difference between the predicted probabilities and the actual labels.

---

## 6. Gradient Descent and Optimization

Gradient descent is used to find the optimal parameters \( heta \) by iteratively adjusting them in the direction that reduces the cost function. In **scikit-learn**, optimization is handled internally using algorithms like **L-BFGS**.

---

## 7. Regularization in Logistic Regression

Regularization helps prevent **overfitting** by penalizing large weights:

- **L1 (Lasso)**: Adds an absolute value penalty on the weights.
- **L2 (Ridge)**: Adds a squared value penalty on the weights.

---

## 8. Multiclass Classification (Optional)

Logistic regression can be extended to handle multiclass problems using strategies like **one-vs-rest (OvR)** or **softmax regression** for multi-class classification.

---

# Practical Implementation of Logistic Regression with Scikit-learn

## Example Code:

```python
# Step 1: Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Step 2: Load the dataset
data = load_breast_cancer()
X = data['data']
y = data['target']

# Step 3: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Create and train the logistic regression model
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Step 7: Print a detailed classification report
print(classification_report(y_test, y_pred))

# Step 8: Predict probabilities (Optional)
probabilities = model.predict_proba(X_test)
print(probabilities[:5])
```

## Explanation of Code:

- **Train-test split**: The dataset is divided into training and test sets to evaluate the model on unseen data.
- **Model fitting**: The logistic regression model is trained on the training set.
- **Prediction and evaluation**: The model is used to predict test labels, and its accuracy and classification performance are evaluated.
- **Probability predictions**: Optionally, you can use `predict_proba` to get the predicted probabilities for each class.

---

## 9. Pros and Cons of Logistic Regression

- **Pros**: Simple, interpretable, works well with linearly separable data.
- **Cons**: Struggles with non-linear relationships (can be extended using techniques like polynomial or kernel transformations).

## 10. When to Use Logistic Regression

- Logistic regression works well for smaller datasets and when the relationship between features and the target is approximately linear.
