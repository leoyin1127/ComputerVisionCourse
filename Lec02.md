## Lecture 2: 

### A Data-Driven Strategy for Image Classification

1. **Assemble** a dataset of images paired with labels.
2. **Develop** a classification model using machine learning techniques.
3. **Evaluate** the model's performance on unseen images.

Essential Functions:
- `train(images, labels)` for training the model with the dataset.
- `predict(model, test_images)` for predicting labels of new images.

#### Nearest Neighbor Classification

Dataset: CIFAR10

- **Distance Metric** to compare images:
    - **L1 Distance (Manhattan Distance)**: Defined as \(d_1(I_1,I_2)=\sum_p\left|I_1^p-I_2^p\right|\).

``` python
import numpy as np

class NearestNeighbor:
  def __init__(self):
    pass
  
  def train(self, X, y):
    # X is N x D where each row is an example. Y is 1-dimensional of size N.
    # The nearest neighbor classifier simply remembers all the training data
    self.Xtr = X
    self.ytr = y
    
  def predict(self, X):
    # X is N x D where each row is an example we wish to predict label for
    num_test = X.shape[0]
    # Ensure the output type matches the input type
    Ypred = np.zeros(num_test, dtype=self.ytr.dtype)
    
    # Loop over all test rows
    for i in range(num_test):
      # Find the nearest training image to the i-th test image
      # using the L1 distance (sum of absolute value differences)
      distances = np.sum(np.abs(self.Xtr - X[i,:]), axis=1)
      min_index = np.argmin(distances) # Get the index with smallest distance
      Ypred[i] = self.ytr[min_index] # Predict the label of the nearest example
    
    return Ypred

```

Complexities:
- Training complexity: O(1)
- Prediction complexity: O(N)
- Goal: Optimize prediction speed. Training efficiency is a secondary concern.

#### K-Nearest Neighbors (K-NN)

- Strategy: Use a **majority vote** from the **K closest points**.
- **Distance Metrics**:
    - L1 (Manhattan) distance
    - L2 (Euclidean) distance

Hyperparameters:
- Choosing K and the distance metric are problem-dependent and cannot be learned from the data.
- Hyperparameters should be tuned on a validation set, not the test set.

``` python
# Assume we have Xtr_rows, Ytr, Xte_rows, Yte as before
# Xtr_rows is 50,000 x 3072 matrix
Xval_rows = Xtr_rows[:1000, :] # take first 1000 for validation
Yval = Ytr[:1000]
Xtr_rows = Xtr_rows[1000:, :] # keep last 49,000 for train
Ytr = Ytr[1000:]

# Find hyperparameters that work best on the validation set
validation_accuracies = []
for k in [1, 3, 5, 10, 20, 50, 100]:
    # Use a particular value of k and evaluation on validation data
    nn = NearestNeighbor()
    nn.train(Xtr_rows, Ytr)
    # Here we assume a modified NearestNeighbor class that can take a k as input
    Yval_predict = nn.predict(Xval_rows, k=k)
    acc = np.mean(Yval_predict == Yval)
    print(f'accuracy: {acc}')
    
    # Keep track of what works on the validation set
    validation_accuracies.append((k, acc))

```

Challenges with Nearest Neighbors:
- Slow at prediction time.
- Pixel-based distance metrics may not reflect true similarities.
- High-dimensional spaces (like images) introduce complexities.

#### Linear Classification

Parametric Approach:
- Equation: \(f(x,W) = Wx + b\)
    - \(x\): Input data.
    - \(W\): Weight matrix.
    - \(b\): Bias vector.

Considerations:
- Linear classifiers can be seen as performing template matching.
- A common practice is to append a bias dimension to \(x\) to integrate both \(W\) and \(b\) into a single matrix, simplifying the score function to \(f(x,W) = Wx\).

Data Preprocessing:
- Normalize pixel values from [0,255] to [-1,1].
