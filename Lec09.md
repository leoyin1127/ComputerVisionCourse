## Lecture 9 Summary

### Network Architecture Details

#### Common Metrics

* **Memory Calculation**:
  * Total memory (KB) = (total output elements) * (element size in bytes) / 1024
  * For 32-bit elements: = (C * H' * W') * 4 / 1024

* **Parameters Calculation**:
  * Total parameters = weights + biases
  * = (C' * C * H * W) + C'

* **Floating Point Operations (FLOPs)**:
  * FLOPs = output elements * operations per element
  * = (C' * H' * W') * (C * H * W)

### Pioneering Neural Network Architectures

#### AlexNet (2012)

* Deployment on two GPUs
* Predominant memory consumption in initial convolutional layers
* Majority of parameters located in fully-connected (fc) layers
* Highest number of FLOPs found in convolutional layers

#### ZFNet (2013)

* Architecture similar to AlexNet, refined through experimental iterations

#### VGG (2014)

* **Design Principles**:
  * Use 3x3 convolutions with stride 1 and padding 1
  * Employ 2x2 max pooling with stride 2
  * Double the number of channels post-pooling

* **Structure**:
  * Composed of 5 stages, each concluding with a pooling layer:
    1. Two convolutional layers followed by pooling
    2. Two convolutional layers followed by pooling
    3. Two convolutional layers followed by pooling
    4. Three (or four in VGG19) convolutional layers followed by pooling
    5. Three (or four in VGG19) convolutional layers followed by pooling

* **Efficiency**:
  * Two sequential 3x3 convolutions offer the receptive field of a 5x5 convolution but with fewer parameters and reduced computational need.
  * ReLU activations introduced between the 3x3 convolutions add non-linearity.

#### GoogLeNet (2014)

* **Efficiency Innovations**:
  * Initial stem network aggressively reduces input dimensions.

* **Inception Module**:
  * Consists of parallel branches with different kernel sizes, minimizing the need for pre-setting hyperparameters.
  * Incorporates 1x1 convolutions as bottlenecks before more computationally expensive operations.

* **Classification**:
  * Utilizes Global Average Pooling for dimensionality reduction followed by a single linear layer for class prediction.
  * Auxiliary classifiers were rendered obsolete by subsequent Batch Normalization techniques.

#### ResNet (2015)

* **Problem and Solution**:
  * Addressed the degradation problem in deeper networks by introducing skip connections that facilitate identity mappings.

* **Residual Blocks**:
  * Basic Block: Two 3x3 convolutions.
  * Bottleneck Block: A sequence of 1x1, 3x3, and 1x1 convolutions.
  * Pre-activation Block: Incorporates ReLU inside residuals to potentially zero-out convolutional weights and achieve true identity function.

* **Structure**:
  * Similar to VGG, the network is segmented into stages; the first block of each stage doubles the channel count and halves the resolution via a stride-2 convolution.

* **Enhancements**:
  * Employs a stem similar to GoogLeNet for initial input downsampling.
  * Global Average Pooling is used consistently across models.
  * Introduced variations like ResNeXt, which utilizes group convolutions for further efficiency.

### Subsequent Innovations in Network Design

* **2017**: SENet (Squeeze-and-Excitation Networks) improved channel interdependencies.
* **Recent Trends**:
  * Densely Connected Networks
  * Efficient architectures like MobileNets and ShuffleNet
  * Advances in Neural Architecture Search to automate optimal model design.
