## Introduction

1. The authors highlight that stacking layers in a deep neural networks in the vanilla manner degrades network performance after a saturation point. It was also observed that the training error stops converging at a point higher than shallow networks as network depth is increased, indicating that the degradation is not due to overfitting. This is counter-intuitive, as additional parameters and non-linearities usually cause a model to be easy to overfit.![[training_plain_vs_resnet_optimized.png]]
2. The authors briefly discuss the problem of vanishing and exploding gradients in deep neural networks and state that it has been largely addressed by the research community with better normalized weight initialisation strategies and layer normalisation methods, such as Batch Norm. They claim that in ResNet, which uses Batch Normalisation, the flow of gradients was stable and they did not vanish or explode during training.
3. They hypothesise that as the weights in each layer are initialised near zero, optimising the network towards learning complex transformations gets harder with additional depth, especially during the initial training phase. They reformulate the problem by stating that if additional layers are added to a shallow network, the resulting deep network should do at least as well as it's shallow counterpart by learning the identity function in the additional layers, but is unable to do so with only SGD. 
4. To address this, the network is encouraged to learn the residual function, $F(x)=H(x)-x$, rather than the entire output function $H(x)$. This is based on the idea that each layer learns are relatively small transformation to be applied to the input signal, and does not deviate too far from the identity function. If the identity function is the optimal transformation for that layer, it is easier to drive $F(x)\rightarrow0$, than to learn $H(x)=x$ in the original formulation. In practice, this is done by using skip connections, which preserve information by carrying the unmodified input signal containing low-level features over to deeper layers, encouraging their reuse and potentially leading to more efficient feature extraction.
5. Later research on shortcut connections suggests that the skip connections allow the network to dynamically choose which transformations to use for each input, creating many different paths of different lengths through the network. This was shown empirically by removing layers during test time and observing little effect on performance. They also allow an alternate, more direct path for gradients to flow backwards, thus addressing the vanishing gradients problem and allowing very deep networks to be trained very effectively. This can be seen as a form of implicit regularisation.
### Optimisation Landscape
1. Some research suggests that skip connections make the optimisation landscape of very deep networks smoother, making it easier for gradient descent algorithms to find good solutions.
## Data Processing
##### Train-time Augmentations

| ResNet Paper                               | PyTorch                                    |
| ------------------------------------------ | ------------------------------------------ |
| RandomShortestSide(256, 480)               | Resize(256, BILINEAR)                      |
| RandomCrop(224)                            | CenterCrop(224)                            |
| ToDtype(torch.float32, scale = True)       | ToDtype(torch.float, scale = True)         |
| Normalize($\mu_{train}$, $\sigma_{train}$) | Normalize($\mu_{train}$, $\sigma_{train}$) |
| ColorJitter with PCA (AlexNet)             |                                            |
##### Test-time Augmentations
| ResNet Paper (Multi-Crop Evaluation)          |
| --------------------------------------------- |
| Resize(Q), Q  $\in {224, 256, 384, 480, 640}$ |
| RandomHorizontalFlip(0.5)                     |
| FiveCrop(224)                                 |
## Architecture
1. A building block is defined as $y = \mathcal F(x , {W}_{i}) + W_{s}x$, where ${W}_{s}$ is an optional linear projection applied to the input to match dimensions with the input transformed by the residual transformation ($\mathcal F(x, W_i)$, which can represent applying multiple convolution layers to the input.), as they are added together in an element-wise manner.
2. The authors present two major implementations of this building block, the residual block and the bottleneck block, as shown in the figure. In the bottleneck block, to solve the dimension mismatch problem, the authors compare three ways to perform the projection denoted by $W_s$, by zero padding the dimensions resulting in parameter free connections, by using pointwise convolutional blocks wherever increased dimensions are needed, introducing a few parameters and by replacing all shortcut connections with pointwise convolution, introducing many more parameters. It is observed that all three options perform better than plain connections, but only marginally better than each other. In the end, the authors prefer the first and second options, as the increase in model complexity due to the third option, even though it performs the best, seems unnecessary.
3. The authors choose the first convolutional layer with a 7x7 kernel size (? receptive field vs efficiency ?), and a stride of 2, followed by the only max pooling operation in the network, with a kernel size of 3x3 and stride of 2. Spatial downsampling in the network is performed by using stride 2 convolutions, not max pooling. This encoder is then followed by global average pooling and a fully connected layer towards the end. The authors attribute the stability of the training process to the use of Batch Norm in the convolutional blocks.
## Hyperparameters
| Model Name   | Parameter             | Value                                                                                |
| ------------ | --------------------- | ------------------------------------------------------------------------------------ |
| ResNet, 2015 | Optimizer             | `SGD(lr = 0.1, momentum = 0.9, weight_decay = 1e-4)`                                 |
|              | Loss Function         | `Softmax + NLLLoss (multinomial logistic regression)`                                |
|              | Batch Size            | `256`                                                                                |
|              | LR Scheduler          | `ReduceLROnPlateau(factor = 0.1)`                                                    |
|              | Weight Initialization | He Initialization, as described in Delving Deep into Rectifiers, He et al, ICCV 2015 |
|              | Iterations (Steps)    | `60,000`                                                                             |
## Quantitative Evaluation