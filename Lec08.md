## Lecture 8 Summary

#### Hardware and Software Overview

##### GPUs: Specialized in Parallel Processing

- **Core Comparison**: GPUs contain more cores than CPUs, but each core operates at a slower rate. This design is optimized for parallel tasks like matrix multiplication.
- **CUDA**: A programming model that enables dramatic increases in computing performance by harnessing the power of the GPU.
- **Streaming Multiprocessors**: These include FP32 cores and specialized Tensor Cores that are adept at handling 4x4 matrix operations.

##### Pytorch: A Versatile Machine Learning Framework

###### Core Components

- **Tensor**: The primary data structure in Pytorch used for storing and manipulating data across a variety of machine learning models.
- **Modules**: Defined as subclasses of `torch.nn.Module`, allowing for organized and modularized model creation. Example:

  ```python
  class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
      super(TwoLayerNet, self).__init__()
      # Layer definition
    def forward(self, x):
      # Implementation of the forward pass
  ```

  Sequential construction is also supported:

  ```python
  model = torch.nn.Sequential()
  ```

###### Autograd System

- **Gradient Requirements**: Tensors for which gradients must be calculated are marked with `Requires_grad = True`.
- **Backpropagation**: Implemented by calling `Loss.backward()`, which automatically computes the backward pass.
- **Gradient Accumulation**: Gradients are accumulated into tensors like `w1.grad` and `w2.grad`; the computational graph is then discarded.
- **Zeroing Gradients**: To reset gradients, Pytorch uses:

  ```python
  w1.grad.zero_()
  w2.grad.zero_()
  ```

- **Graph-Free Updates**: Updates without tracking through the computation graph:

  ```python
  with torch.no_grad():
    for param in model.parameters():
      param -= learning_rate * param.grad
  ```

- **Optimization Steps**: Simplifying parameter updates with optimizers:

  ```python
  optimizer.step()
  optimizer.zero_grad()
  ```

- **Custom Autograd Functions**: Extending Pytorch's autograd by defining custom functions:

  ```python
  class Sigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
      # forward logic here
    @staticmethod
    def backward(ctx, grad_output):
      # backward logic here
  def sigmoid(x):
    return Sigmoid.apply(x)
  ```

###### Computation Graph Dynamics

- **Dynamic Graphs**: Pytorch's ability to build and compute graphs simultaneously facilitates the use of regular Python control flows, which is particularly beneficial for models like RNNs that depend on sequential data.

- **Static Graphs**: These are built once and reused, enhancing efficiency and performance, especially in production environments where the input does not vary:

  ```python
  @torch.jit.script
  ```

  Comparing Graph Types:

  | Feature       | Static                                   | Dynamic                                              |
  | ------------- | ---------------------------------------- | ---------------------------------------------------- |
  | Optimization  | Pre-execution optimization is possible   | Not applicable                                       |
  | Serialization | Can be serialized and deployed code-free | Requires code presence for deployment                |
  | Debugging     | More complex due to abstraction          | Simpler as code directly represents runtime behavior |

###### Parallel Computing

- **Data Parallelism**: Leveraging multiple GPUs to accelerate training processes via:
  - `nn.DataParallel`
  - `nn.DistributedDataParallel`

##### TensorFlow Framework Evolution

- **TF 1.0**: Utilizes static computation graphs that are defined once and executed multiple times.
- **TF 2.0**: Introduces more flexibility with the `@tf.function` for creating static graphs within a dynamic programming environment.
- **Keras**: Facilitates rapid prototyping and research through a high-level API that integrates seamlessly with TensorFlow.

##### Additional Resources

- **Pre-trained Models**: Accessible through `torchvision`.
- **TensorBoard**: Enables detailed visualization and tracking of model training and performance metrics.
