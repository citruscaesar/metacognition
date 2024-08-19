## Introduction
* A very large image dataset is required to capture the variance in real world images to solve the complex object recognition task. Projects like ImageNet and LabelMe present such datasets, but even those datasets with millions of images don't completely capture the entire variability. At the time very carefully engineered feature extractors like SIFT, SURF and Bag of Words (Images) were used to convert real life images to an alternate representation, which were then used as input to classical ML approaches to classify images, instead of raw pixel data.
* Convolutional Neural Networks were shown to be successful in classifying a low resolution OCR dataset (MNIST) by LeCun et. al, without human engineered feature extractors by employing a learning algorithm.  ConvNets encode a strong prior belief about the nature of images, as seen in the assumptions about images stationarity of statistics and locality of pixel dependencies. %%expand about these properties.%%
* Krizhevsky et. al.[^AlexNet,2012] expanded on the work of LeCun et al, and trained a ConvNet with 5 convolutional and 3 fully connected layers (henceforth called AlexNet) on 2x RTX 580s using an efficient GPU implementation of 2D Convolution (+ related operations) and a Trainer . The network ended up winning the ILSVRC-2012 challenge, ushering in a new era of Computer Vision.
## Data
* The authors of AlexNet used the ILSVRC-2010 subset of ImageNet-1K which is the last one to have a publicly available testing set to experiment with.
##### Pre-processing Steps
| Common                       | PyTorch                                    |
| ---------------------------- | ------------------------------------------ |
| Resize(256)                  | Resize(256, BILINEAR)                      |
| CenterCrop(256)              | CenterCrop(224)                            |
| Subtract Mean($\mu_{train}$) | ToDtype(torch.float, scale = True)         |
|                              | Normalize($\mu_{train}$, $\sigma_{train}$) |
* CenterCrop extracts square images from rectangular images without changing their aspect ratio
* $\mu_{train} = [0.485, 0.456, 0.406],\;\sigma_{train} = [0.229, 0.224, 0.225]$
##### Train-time Augmentations
| AlexNet Papers            | PyTorch                       |
| ------------------------- | ----------------------------- |
| RandomCrop(224)           | RandomHorizontalFlip(0.5)<br> |
| RandomHorizontalFlip(0.5) |                               |
| ColourJitter with PCA     |                               |
* ColorJitter on $\mathcal{D}_{train}$: $Pixel\;Intensity, I_{x,y}=[I^{R}_{x,y}, I^{G}_{x,y}, I^{B}_{x,y}]\mathrel{\raise{0.19ex}{\scriptstyle+}}=[p_1, p_2, p_3][\alpha_1\lambda_1, \alpha_2\lambda_2, \alpha_3\lambda_3]^T\;where,\, p_i: i^{th} eigenvector,\lambda_i: i^{th} eigenvalue, \alpha_i \sim \mathcal{N}(0, 0.1)$
##### Test-time Augmentations
| AlexNet Two Tower         | AlexNet One Tower |
| ------------------------- | ----------------- |
| FiveCrop(224)             | CenterCrop(224)   |
| RandomHorizontalFlip(0.5) |                   |
* Average of all softmax outputs is used for prediction

## Architecture
\
<center> Fig1: Two "column" version of the AlexNet architecture, from <em> Imagenet Classification using Deep Neural Networks, Alex Krizhevsky et. al., 2012 </em> It features model parallelism across 2xGPUs</center>

1. ReLU is shown to converge much faster than Sigmoid or Tanh (5 vs 35 epochs to reach 0.25 error) when tested on CIFAR10 using a 4 layer ConvNet. Tanh was reported to work well with Contrast Normalization, which is not used here.
2. Local Response Normalization: It is applied after ReLU and even though ReLU does not saturate, the authors of AlexNet claimed that it helps the model generalize better. This is inspired by real neurons, where the values of the feature maps due to each kernel is inhibited from growing too much by creating a competition for larger activations. The response normalized activation of the neuron at the $i^{th}$ kernel map at the $(x,y)^{th}$ spatial position is given by:
$$b_{x,y}^i =\frac {a_{x,y}^i}{(k+\alpha \sum_{j=max(0, i-n/2)}^{min(N-1, i+n/2)}({a_{x,y}^i})^2)^{\beta}},\;k=2,\:N=5,\:\alpha=10^{-4},\:\beta=0.75\;$$
 <center>Fig 0: LRN is caculated by dividing the activation of each neuron by the average activation across at least N feature maps for each feature map for each spatial location.</center>
 
5. `MaxPool2d(kernel_size = 3, stride = 2)` was used, notably with overlapping pooling windows since stride < kernel size. Authors of AlexNet claimed it helps to both down-sample the feature maps and regularize the model slightly.

6. `Dropout(p=0.5)` was used on both fc6 and fc7 to prevent overfitting.

7. Model parallelism was used to split the weights of the ConvNet across 2 RTX 580s, since each GPU only had 3GB VRAM. Later (Alex Krizhevsky, 2014) presented an alternate implementation using Data Parallelism for the Conv layers (#params in conv layers) and Model Parallelism for the Fully Connected Layers (#params in fc layers)


<center> Fig2: Single "column" version of the AlexNet architecture, from <em> One weird trick for parallelizing convolutional neural networks, Alex Krizhevsky, 2014 </em>. It features data parallelism for the convolutional layers, and model parallelism for the fully connected layers.</center>

<center> Fig3: The "improved" ZFNet architecture from <em>Visualizing and Understanding Convolutional Neural Networks, Matthew Zieler et. al., 2013</em></center>

## Hyperparameters

| Model Name                          | Parameter             | Value                                                                    |
| ----------------------------------- | --------------------- | ------------------------------------------------------------------------ |
| AlexNet, 2012 [^alexnet,2012]       | Optimizer             | `SGD(lr = 0.01, momentum = 0.9, weight_decay = 5e-4)`                    |
|                                     | Loss Function         | `Softmax + NLLLoss (multinomial logistic regression)`                    |
|                                     | Batch Size            | `128`                                                                    |
|                                     | LR Scheduler          | `ReduceLROnPlateau(factor = 0.1)`                                        |
|                                     | Weight Initialization | $W\sim\mathcal{N}(0, 0.01),\:\mathcal{b}_{1,3}=0,\:\mathcal{b}_{rest}=1$ |
| AlexNet, 2014 [^alexnet,2014]       | Loss Function         | `CrossEntropyLoss`                                                       |
|                                     | LR Scheduler          | `MultiStepLR(milestones = [22, 45, 68], gamma = pow(250, -1/3))`         |
|                                     | Batch Size            | `{128, 256, 512, 1024} with data parallelism`                            |
| AlexNet, Pytorch [^alexnet,pytorch] | Optimizer             | `SGD(lr = 0.01, momentum = 0.9, weight_decay = 1e-4)`                    |
| 61,100,840 params                   | Loss Function         | `CrossEntropyLoss`                                                       |
| 0.71 GFLOPS                         | Epochs                | `90`                                                                     |
| 233.1 MB                            | Batch Size            | `32 with data parallelism on 8xV100`                                     |
|                                     | LRScheduler           | `StepLR(step_size = 30, gamma = 0.1)`                                    |
| ZFNet                               |                       |                                                                          |
## Qualitative Evaluation

1. Plotting the kernels learnt by the conv1 layer shows that it learns  a "variety of frequency and orientation-selective kernels, as well as various coloured blobs". Interestingly, one GPU learns colour agnostic filters and the other one learns colour-specific filters, irrespective of how the weights were initialized for each GPU.
2. The model also displays translational invariance, as it is able to correctly classify off-centre objects in the images.
3. The authors show that the top-5 predictions are mostly meaningful, such as cats and leopards. But in case of ambiguity with multiple labels being in the same image, like dogs and cherries, the model finds it difficult to choose the focus of the image.
4. The authors state that the Euclidean Distance b/w the fc7 activations $\in \mathbb{R}^{4096}$ of images belonging to the same class will be smaller than those of semantically different classes, invariant to the pose or orientation, contrary to the pixel level representation. Using autoencoders to compress this features space should result in faster and more efficient image retrieval methods since calculating the L2 norm of vectors in a 4096 dimensional space is slow. They also say that this is better than training autoencoders on raw pixel data without the labels, as they are not able to learn meaningful semantic relationships between images, and would end up grouping images with similar edge patterns together or something.
## Quantitative Evaluation

| Model Name (Variant)                                             | Dataset                                     | Top-1 Accuracy (%) | Top-5 Accuracy (%) |
| ---------------------------------------------------------------- | ------------------------------------------- | ------------------ | ------------------ |
| AlexNet (1x, + a conv6 layer)                                    | ImageNet-10k, 2009 (Test, 50% random split) | 32.6               | 59.1               |
| AlexNet (1x)                                                     | ILSVRC-2010 (Test)                          | 62.5               | 83                 |
| AlexNet (1x)                                                     | ILSVRC-2012 (Val/Test)                      | 59.3 / -           | 81.8 / 81.8        |
| AlexNet (5x, Averaged)                                           | ILSVRC-2012 (Val/Test)                      | 61.9 / -           | 83.6 / 83.7        |
| AlexNet (1x, Pretrained on Imagenet-22k, 2011)                   | ILSVRC-2012 (Val/Test)                      | 61 / -             | 83.4 / -<br>       |
| AlexNet (7x, 2x Pretrained on Imagenet 22k, 2011, 5x from above) | ILSVRC-2012 (Val/Test)                      | 63.3 / -           | 84.6 / 84.7        |
| AlexNet : One Tower (1x)                                         | ILSVRC-2012 (Val)                           | 57.73              | -                  |
| AlexNet : PyTorch (1x)                                           | ILSVRC-2012 (Val or Test)                   | 56.522             | 79.066             |
## Discussion
##### AlexNet 2012
> Our results show that a large, deep convolutional neural network is capable of achieving record- breaking results on a highly challenging dataset using purely supervised learning. It is notable that our network’s performance degrades if a single convolutional layer is removed. For example, removing any of the middle layers results in a loss of about 2% for the top-1 performance of the network. So the depth really is important for achieving our results. To simplify our experiments, we did not use any unsupervised pre-training even though we expect that it will help, especially if we obtain enough computational power to significantly increase the size of the network without obtaining a corresponding increase in the amount of labeled data. Thus far, our results have improved as we have made our network larger and trained it longer but we still have many orders of magnitude to go in order to match the infero-temporal pathway of the human visual system. Ultimately we would like to use very large and deep convolutional nets on video sequences where the temporal structure provides very helpful information that is missing or far less obvious in static images.
## References and Footnotes

[^AlexNet,2012]: [[Imagenet Classification with Deep Convolutional Neural Networks, Alex Krizhevsky et. al, 2012.pdf]]
[^AlexNet,2014]: [[One weird trick for parallelizing convolutional neural networks, Alex Krizhevsky, 2014.pdf]]
[^AlexNet,Pytorch]: [https://pytorch.org/vision/main/models/generated/torchvision.models.alexnet.html]
[^ZFNet,2013]: [[Visualizing and Understanding Convolutional Networks, Matthew Zeiler et al., 2013.pdf]]