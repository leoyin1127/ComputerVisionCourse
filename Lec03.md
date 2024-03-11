## Lecture 3 

### Evaluating Model Performance with Loss Functions

- **Objective of Loss Functions**: To quantify the model's accuracy.

- **Calculating Loss Over a Dataset**: The average loss is determined by \(L = \frac{1}{N} \sum_i L_i(f(x_i, W), y_i)\).

- **Details on Multiclass SVM Loss**:
  - Each image-label pair \((x_i, y_i)\) consists of an image \(x_i\) and its corresponding label \(y_i\).
  
  - Introducing a shorthand for the scores vector: \(s_i = f(x_i, W)\).
  
  - The SVM Loss formula is represented as \(L_i = \sum_{j \neq y_i} \max(0, s_j - s_{y_i} + 1)\), also known as the hinge loss. This formula may extend to squared hinge loss.
  
  - The aim is to have the correct class score \(s_{y_i}\) exceed the scores of incorrect classes by at least a margin Δ (chosen to be 1). Initially, assuming scores near zero, the loss approximates the count of classes minus one.
  
  - The method of summing or averaging losses does not alter the overall scale.
  
  - There are multiple possible weight configurations that satisfy the model criteria, such as \(W\), \(2W\), \(3W\), etc.
  
    ``` python
    def compute_loss_vectorized(x, y, W):
    score_matrix = W.dot(x)
    delta = 1.0
    correct_class_score = score_matrix[y]
    margin = np.maximum(0, score_matrix - correct_class_score + delta)
    margin[y] = 0
    loss_value = np.sum(margin)
    return loss_value
    ```
  
- **Inclusion of Regularization**:
  - Promotes model simplicity to prevent overfitting, in line with Occam's razor principle.
  
  - The regularization strength, symbolized by \(\lambda\), is a critical hyperparameter.
  
  - Integrating with gradient descent leads to weight decay.
  
  - Regularization strategies include:
    - **L2 Regularization**: Favors smaller, distributed weights.
    - **L1 Regularization**: Leads to a sparse weight matrix.
    - **Elastic Net**: Combines L1 and L2 regularization effects.
  
- **Fundamentals of the Softmax Classifier**:
  - Models the probability for class \(k\) given \(x_i\) as \(P(Y=k | X=x_i) = \frac{e^{s_k}}{\sum_j e^{s_j}}\).
  
  - Aims to minimize the negative log likelihood of the correct class, aligning with Maximum Likelihood Estimation (MLE).
  
  - Positions regularization as a Gaussian prior on \(W\), transitioning to Maximum a Posteriori (MAP) estimation.
  
  - Initial loss expectations start at \(\log(C)\), with possible values ranging from 0 to infinity.
  
  - **Addressing Numerical Stability**: Implements score vector normalization to prevent division by large numbers.
  
    ``` python
    def adjust_scores_for_stability(f):
    f_adjusted = f - np.max(f)
    probabilities = np.exp(f_adjusted) / np.sum(np.exp(f_adjusted))
    return probabilities
    ```
  
  - **Cross-entropy Loss**: Focuses on minimizing the discrepancy between true and estimated probability distributions, enhancing the accuracy for the correct class.

### Insights into Supervised Learning

### Optimization Approaches

- **Convex Optimization Insights**: For further reading, check [Stanford's Convex Optimization book](https://stanford.edu/~boyd/cvxbook/).

- **Exploration with Random Search**: Understands its boundaries and applications.

- **Gradient Orientation**: Identifies the negative gradient direction as the path of steepest descent.
  
  - Discusses the distinctions between numerical and analytical gradients in terms of speed, accuracy, and susceptibility to errors.
  
  - **Validating Gradient Accuracy**:
    - Emphasizes the importance of comparing analytical gradients with numerical approximations to confirm correctness.
    
    ``` python
    def compute_numerical_gradient(f, x):
    fx = f(x)  # Evaluate function value at original point
    grad = np.zeros_like(x)
    h = 0.00001
    
    # Iterating over all dimensions of x to compute gradient
    for index in range(x.size):
        old_value = x[index]
        x[index] = old_value + h
        fxh = f(x)  # Evaluate f(x + h)
        x[index] = old_value  # Reset to previous value
        
        # Calculate partial derivative
        grad[index] = (fxh - fx) / h
        
    return grad
    ```

- **Implementing Gradient Descent**:
  - Formulates loss as \(L(W) = \frac{1}{N} \sum^N_{i=1} \nabla W L_i(x_i, y_i, W) + \lambda \nabla_W R(W)\).
  
   ``` python
   def perform_gradient_descent(loss_function, data, weights, learning_rate):
    while True:
        gradient = evaluate_gradient(loss_function, data, weights)
        weights -= learning_rate * gradient  # Update weights
  ```
  
  - Highlights the critical role of learning rate selection and introduces minibatch processing for efficiency.

    ``` python
    def perform_minibatch_gradient_descent(loss_function, data, weights, learning_rate, batch_size=256):
    while True:
        data_batch = sample_training_data(data, batch_size)
        gradient = evaluate_gradient(loss_function, data_batch, weights)
        weights -= learning_rate * gradient  # Update weights
    ```
### Image Feature Enhancement Techniques

- **Advantage of Effective Features**: Demonstrates superiority over raw pixel input.
  
- Incorporates techniques such as **Color Histograms**, **Histogram of Oriented Gradients (HOG)**, **Bag of Words**, and the application of **Convolutional Neural Networks (ConvNets)** to learn features directly from data.
