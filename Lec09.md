#### Net Architecture

* Some indexes:

  * Memory: KB = (num of output elements) * (bytes per element) / 1024

    ​                       = (C * H' * W') * 4 [for 32-bit floating point] / 1024

  * Params: number of weights = (weight shape) + (bias shape)

    ​                                                   = C' * C * H * W + C'

  * Number of floating point operations (flop: multiply + add) 

    ​	= (number of output elements) * (ops per output element)

    ​	= (C' * H' * W') * (C * H * W)

##### AlexNet (2012) 

* Model split over two GPUs
* Most of the memory usage is in the early convolution layers
* Nearly all of the parameters are in the fc layers
* Most floating-point ops occur in the convolution layers

##### ZFNet (2013)

* Both the architecture of AlexNet and ZFNet are set by trials and errors (hand-designed)

##### VGG (2014)

* VGG Design rules:
  * All convolutions are 3x3 stride 1 pad 1
  * All max pools are 2x2 stride 2
  * After pooling, double the number of channels
* Have 5 convolution stages:
  1. conv-conv-pool
  2. conv-conv-pool
  3. conv-conv-pool
  4. conv-conv-conv-[conv]-pool
  5. conv-conv-conv-[conv]-pool (VGG19 has 4 convolutions in stages 4 and 5)

* Two 3x3 convolutions have the same receptive field as a single 5x5 convolution, but have fewer parameters and take less computation. Additionally, two 3x3 convolutions can incorporate more non-linearity by inserting ReLU between them.
* Convolution layers at each spatial resolution take the same amount of computation by doubling channels and halving its spatial resolution

##### GoogLeNet (2014)

* Innovations for efficiency
* Stem network at the start aggressively downsamples the input

* Inception module: local unit with parallel branches which is repeated many times throughout the network
  * Tries all sizes of kernels instead of setting it as a hyperparameter
  * Uses 1x1 bottleneck layers to reduce channel dimension before expensive convolutions

* Uses Global Average Pooling to collapse spatial dimensions, and one linear layer to produce class scores

* Auxiliary Classifiers: no longer needed after Batch Normalization (BN)

##### ResNet (2015)

* Problem: Deeper networks perform worse than shallow ones
* Hypothesis: deeper networks are harder to optimize, and don't learn identity functions to emulate shallow models
* Solution: just copying the learned layers from the shallower model and setting additional layers to identity mapping

* Residual block:
  * Basic block: two 3x3 convolutions
  * Bottleneck block: 1x1 convolution + 3x3 convolution + 1x1 convolution
  * Pre-activation block: ReLU inside residual → can learn true identity function by setting convolution weights to zero
* Network is divided into stages like VGG: the first block of each stage halves the resolution (with stride-2 convolution) and doubles the number of channels

* Uses the same aggressive stem as GoogLeNet to downsample the input 4x before applying residual blocks
* Uses Global Average Pooling
* Improving ResNets: ResNeXt (group convolution)
