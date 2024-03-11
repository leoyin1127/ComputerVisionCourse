## Lecture 6 Summary

### Activation Functions Overview

#### Different Activation Functions and Their Characteristics

- **Sigmoid Function**:
  - Formula: \(\sigma(x) = 1 / (1 + e^{-x})\)
  - Outputs range between [0,1], interpretable as neuron firing rates.
  - Issues include gradient vanishing, non-zero-centered outputs, and computational expense.

- **Tanh Function**:
  - Outputs range between [-1,1], acting as a scaled sigmoid.
  - Zero-centered, but still suffers from gradient vanishing when saturated.

- **ReLU (Rectified Linear Unit)** - **Most Effective**:
  - Formula: \(f(x) = \max(0, x)\)
  - Advantages include non-saturation (in positive range) and computational efficiency.
  - Issues: Non-zero-centered output and gradient death for \(x \leq 0\).

- **Leaky ReLU** - **A Good Variant**:
  - Formula: \(f(x) = \max(0.1x, x)\)
  - Prevents dead neurons by allowing a small, positive gradient when \(x\) is negative.

- **Parametric ReLU**:
  - Allows backpropagation into \(\alpha\), making the small gradient for \(x \leq 0\) learnable.

- **ELU (Exponential Linear Units)**:
  - Offers all ReLU benefits, near-zero mean outputs, and added noise robustness.

- **SELU (Scaled Exponential Linear Units)**:
  - Similar in spirit to ELU with self-normalizing properties.

- **Maxout Neuron** - **Another Good Option**:
  - Generalizes ReLU and Leaky ReLU without saturation or dead units.
  - Doubles the parameter count, which can be a drawback.

#### Data Preprocessing Techniques

- **Zero-centering Data**: Standard practice to improve model convergence.

- **Data Normalization**: Essential for consistent model training performance.

- **PCA and Whitening**: Techniques for data decorrelation and normalization.

``` python
# PCA and Whitening
X -= np.mean(X, axis=0)  # Zero-center the data
cov = np.dot(X.T, X) / X.shape[0]  # Compute covariance matrix
U, S, V = np.linalg.svd(cov)  # Singular Value Decomposition
Xrot = np.dot(X, U)  # Decorrelate the data
Xwhite = Xrot / np.sqrt(S + 1e-5)  # Whiten the data
```

#### Weight Initialization Strategies

- Importance of proper initialization for symmetry breaking and learning efficiency.

- **Xavier and Kaiming Initializations**: Recommended methods for different activation functions to ensure effective gradient flow.

###  Batch Normalization and Learning Process Management

#### Implementing Batch Normalization

- Normalizes layer inputs to be unit Gaussian, improving training dynamics.

- Includes a scaling and shifting step, allowing the model to undo the normalization if beneficial.

#### Strategies for Monitoring and Tuning the Learning Process

- Initial sanity checks with loss levels and overfitting small data subsets.

- Monitoring key metrics during training to guide adjustments and improvements.

- Hyperparameter optimization techniques, including learning rate and regularization tuning.

``` python
# Hyperparameter Optimization
learning_rate = 10 ** np.random.uniform(-6, 1)  # Learning rate search in log space
dropout = np.random.uniform(0, 1)  # Dropout rate search in original scale
```

#### Tips for Hyperparameter Search

- Emphasizing the use of log space for learning rate exploration and the original scale for others like dropout rates.

- The effectiveness of random search over grid search and ensuring search boundaries are appropriately set.
