## Lecture 5 Notes

### Fundamentals of Convolutional Neural Networks

#### Core Components of CNNs

- **Convolution Layer**: Maintains the spatial structure of the input.

- **Filters and Activation Maps**: Filters span the full depth of the input volume, each generating an activation map. The total number of activation maps equals the number of filters.

- **Convolution Operation**: Involves computing the dot product between a filter and local regions of the image. Filters are reshaped into vectors for this computation.

- **Feature Hierarchy**:
  - Early layers capture low-level features like edges.
  - Middle layers detect more complex features such as corners and textured patterns.
  - Later layers identify high-level features.

- **Output Dimension Formula**: The size of the output is calculated as **(N - F) / stride + 1**. When zero-padding is applied to maintain the original input size, the formula adjusts to **(N - F + 2P) / stride + 1**.

- **Parameters**: The layer has **F x F x D x K** weights and **K** biases, where **F** is the filter size, **D** is the depth, and **K** is the number of filters.

### CNN Architecture Components

#### Enhancing CNN with Pooling and Fully Connected Layers

- **5x5 Filters**: Correspond to a 5x5 receptive field.

- **Pooling Layer**:
  - Reduces representation size, making it more compact.
  - Operates spatial downsampling while keeping the depth intact.
  - Commonly uses a stride setting to prevent overlap, with max pooling being a prevalent choice.
  - Output size calculation remains **(N - F) / stride + 1**.
  - Padding is typically not used in pooling layers.
  - The depth dimension is preserved post-pooling.

- **Fully Connected Layer**: Integrates features globally across the input volume.

- **Common CNN Architectures**: Structured as **[(CONV - RELU) * N - POOL] * M - (FC - RELU) * K, SOFTMAX**, where:
  - **N** typically goes up to 5, indicating the repetition of CONV-RELU pairs before a pooling layer.
  - **M** denotes the sequence of CONV-RELU-POOL blocks in the network.
  - **K** ranges from 0 to 2, representing the number of fully connected layers before the softmax output.
