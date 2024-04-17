## Lecture 8 Summary

#### Hardware and software 

##### GPU

* Compared to CPU: more cores, but each core is much slower, and good at doing parallel tasks: Matrix multiplication
* CUDA
* Streaming multiprocessors: FP32 cores, Tensor Core (4x4 matrix)

##### Pytorch

* Tensor
* Autograd
  * ```python
    Requires_grad = True
    ```
  * ```python
    Loss.backward()
    ```
  * Gradients are **accumulated** into w1.grad and w2.grad and the graph is destroyed
  * Set gradients to zero:
    ```python
    w1.grad.zero_()
    w2.grad.zero_()
    ```
  * Tell pytorch not to build a graph for these operations:
    ```python
    with torch.no_grad():
      # gradient descent...
      for param in model.parameters():
        param -= learning_rate * param.grad
    ```
  * Use optimizer to update params and zero gradients:
    ```python
    optimizer.step()
    optimizer.zero_grad()
    ```
  * Can define new functions, but pytorch still creates computation graphs step by step (numerical unstable)
  * Define new autograd operators by subclassing Function, define forward and backward:
    ```python
    class Sigmoid(torch.autograd.Function):
      @staticmethod
      def forward(ctx, input):
        output = 1 / (1 + torch.exp(-input))
        ctx.save_for_backward(output)
        return output
      
      @staticmethod
      def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        grad_input = grad_output * output * (1 - output)
        return grad_input
    
    def sigmoid(x):
      return Sigmoid.apply(x)
    ```
    * Only adds one node to the graph

* Module
  * Define Modules as a torch.nn.Module subclass:
    ```python
    class TwoLayerNet(torch.nn.Module):
      def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        # ...
        
      def forward(self, x):
        # ...
        
    	# no need to define backward - autograd will handle it
    ```
  * Stack multiple instances of the component in a sequential:
    ```python
    model = torch.nn.Sequential()
    ```

* Pretrain Models: torchvision
* tensorboard
* Dynamic Computation Graphs:
  * Building the graph and computing the graph happen at the same time
  * let u use regular Python control flow during the forward pass
  * Applications: model structure that depends on the input (**RNN**)

* Static Computation Graph:
  1. Build computational graph describing our computation
  2. Reuse the same graph on every iteration
    ```python
    @torch.jit.script # python function compiled to a graph when it is defined
    ```

|               | Static                                                                                                       | Dynamic                                                                                |
| ------------- | ------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------- |
| Optimization  | Framework can optimize the graph before it runs                                                              | None                                                                                   |
| Serialization | Once the graph is built, can serialize it and run it without the code (tarin model in Python, deploy in C++) | Graph building and execution are intertwined, so always need to keep code around (RNN) |
| Debugging     | Lots of indirection - can be hard to debug, benchmark, etc                                                   | The code u write is the code that runs. Easy to reason about, debug                    |

* Data parallel
  * nn.DataParallel
  * nn.DistributedDataParallel

##### TensorFlow

* TF 1.0 (Static Graphs)
  * Define computational graph
  * run rhe graph many times
* TF 2.0 (Dynamic Graphs)
  * Static: @tf.function
* Keras
