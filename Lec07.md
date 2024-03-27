## Lecture 7 Summary

#### Gradient Optimization Techniques

- Challenges with Stochastic Gradient Descent (SGD):
  - High variance in the gradient can lead to inefficient convergence paths.
  - Susceptibility to getting stuck at local minima or saddle points due to zero gradients.
  - Noise in gradients due to mini-batch processing.

- Enhancements to SGD:

  - **SGD with Momentum**:
  
    - Formulas: $v_{t+1} = \rho v_t + \nabla f(x_t)$, $x_{t+1} = x_t - \alpha v_{t+1}$.
  
    - Implementation:
      ```python
      vx = 0
      while True:
          dx = compute_gradient(x)
          vx = rho * vx + dx
          x -= learning_rate * vx
      ```
  
    - Momentum acts as a dampener, smoothing the optimization path.
  
  - **SGD with Nesterov Momentum**:
  
    - Formulas: $v_{t+1} = \rho v_t - \alpha \nabla f(x_t + \rho v_t)$, $x_{t+1} = x_t + v_{t+1}$.
  
    - Implementation:
      ```python
      dx = compute_gradient(x)
      old_v = v
      v = rho * v - learning_rate * dx
      x += -rho * old_v + (1 + rho) * v
      ```

- **AdaGrad**:
  
  - Implementation:
    ```python
    grad_squared = 0
    while True:
        dx = compute_gradient(x)
        grad_squared += dx * dx
        x -= learning_rate * dx / (np.sqrt(grad_squared) + 1e-7)
    ```
  
  - Adapts learning rate based on the square of gradients, beneficial for convex problems.

- **RMSProp**:
  
  - Implementation:
    ```python
    grad_squared = 0
    while True:
        dx = compute_gradient(x)
        grad_squared = decay_rate * grad_squared + (1 - decay_rate) * dx * dx
        x -= learning_rate * dx / (np.sqrt(grad_squared) + 1e-7)
    ```

- **Adam Optimizer**:
  
  - Implementation:
    ```python
    first_moment = 0
    second_moment = 0
    for t in range(num_iterations):
        dx = compute_gradient(x)
        first_moment = beta1 * first_moment + (1 - beta1) * dx
        second_moment = beta2 * second_moment + (1 - beta2) * dx * dx
        first_unbias = first_moment / (1 - beta1 ** t)
        second_unbias = second_moment / (1 - beta2 ** t)
        x -= learning_rate * first_unbias / (np.sqrt(second_unbias) + 1e-8)
    ```
  
  - Combines momentum and RMSProp strategies, widely used with default settings of beta1 = 0.9, beta2 = 0.999, and learning_rate = 1e-3 or 5e-4.

- **Learning Rate Strategies**:
  - Various decay methods such as step decay, exponential decay, cosine decay, linear decay, inverse square root, and 1/t decay to adjust the learning rate over time based on performance.
  - A warm-up phase can help mitigate the risk of divergent losses at the start of training.

- **First-Order vs. Second-Order Optimization**:
  - First-order methods utilize gradient information for minimization.
  - Second-order methods leverage both gradient and curvature (Hessian) but are computationally intensive.
  - Quasi-Newton methods, like BFGS and L-BFGS, offer practical alternatives by approximating the Hessian.

- **Model Ensembling Techniques**:
  - Combining multiple models or versions of a model can enhance performance.
  - Techniques include averaging the outputs of different models, using multiple training snapshots, or the moving average of parameters.


#### Strategies for Regularization

- **L1, L2, Maxnorm Regularization**:
  - Techniques to prevent overfitting by penalizing large weights.

- **Dropout**:
  - Randomly sets a portion of neurons to zero during each training forward pass.
  - The dropout rate, often set to 0.5, acts as a hyperparameter.
  - Implementation example:
    ```python
    p = 0.5

    def train_step(X):
        H1 = np.maximum(0, np.dot(W1, X) + b1)
        U1 = np.random.rand(*H1.shape) < p  # Apply dropout mask
        H1 *= U1  # Dropout applied
        H2 = np.maximum(0, np.dot(W2, H1) + b2)
        U2 = np.random.rand(*H2.shape) < p  # Second dropout mask
        H2 *= U2  # Dropout applied
        out = np.dot(W3, H2) + b3

    def predict(X):
        H1 = np.maximum(0, np.dot(W1, X) + b1) * p
        H2 = np.maximum(0, np.dot(W2, H1) + b2) * p
        out = np.dot(W3, H2) + b3
    ```
  - Conceptually, dropout creates an ensemble effect within a single model by generating numerous sub-models.

- **Inverted Dropout**:
  - Scales activations during training so no adjustment is needed at test time.
  - Example:
    ```python
    p = 0.5

    def train_step(X):
        H1 = np.maximum(0, np.dot(W1, X) + b1)
        U1 = (np.random.rand(*H1.shape) < p) / p  # Apply scaled dropout mask
        H1 *= U1  # Scaled dropout applied
        H2 = np.maximum(0, np.dot(W2, H1) + b2)
        U2 = (np.random.rand(*H2.shape) < p) / p  # Second scaled dropout mask
        H2 *= U2  # Scaled dropout applied
        out = np.dot(W3, H2) + b3
    ```

- **Batch Normalization**:
  - Introduces noise/stochasticity during training and averages it out at test time, similar to dropout.

- **Data Augmentation**:
  - Enhances training data variability without changing labels through random transformations such as crops, scales, and color jitters.

- **DropConnect**:
  - An alternative to dropout that randomly zeroes out weights instead of activations.

- **Fractional Max Pooling**:
  - Uses random pooling regions to introduce stochasticity into the network architecture.

- **Stochastic Depth**:
  - Randomly drops entire layers during training to reduce overfitting and computational cost.

- **Cutout/Mixup**:
  - Techniques specifically designed for enhancing generalization in small dataset image classifications.


#### Transfer Learning

* One reason of overfitting is there is not enough data

|                         | **Very similar dataset**           | **Very different dataset**                               |
| ----------------------- | ---------------------------------- | -------------------------------------------------------- |
| **Very little data**    | Use Linear Classifier on top layer | Difficult... Try linear classifier from different stages |
| **Quite a lot of data** | Finetune a few layers              | Finetune a larger number of layer                        |
