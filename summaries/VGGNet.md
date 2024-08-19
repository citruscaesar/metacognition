## Introduction
1. The authors talk about the attempts made towards improving the AlexNet architecture since 2012, such as the use of smaller kernels and strides by the authors of ZFNet and the training and evaluation densely over the whole image and at multiple scales by Sermanet et al, 2014.
2. They study the effects of increasing the depth of the ConvNet architecture by independent of other hyperparameters, which is computationally feasible by using convolutional kernels of size 3x3 exclusively, and arrive at more accurate architectures which achieve SOTA on ILSVRC and other competitions.
## Data
##### Pre-processing Steps
| VGG Paper                                                                          | PyTorch                                        |
| ---------------------------------------------------------------------------------- | ---------------------------------------------- |
| <br>ToDtype(torch.float, scale = True)                                             | Resize(256, BILINEAR)                          |
| <br>Subtract Mean($\mu_{train}$) or <br>Normalize($\mu_{train}$, $\sigma_{train}$) | CenterCrop(224)                                |
|                                                                                    | ToDtype(torch.float, scale = True)             |
|                                                                                    | <br>Normalize($\mu_{train}$, $\sigma_{train}$) |
##### Train-time Augmentations
| Single Scale                   | Multi Scale                    | PyTorch                   |
| ------------------------------ | ------------------------------ | ------------------------- |
| Resize(256) or Resize(384)     | RandomShortestSide(256, 512)   | RandomHorizontalFlip(0.5) |
| RandomCrop(224)                | RandomCrop(224)                |                           |
| RandomHorizontalFlip(0.5)      | RandomHorizontalFlip(0.5)      |                           |
| ColorJitter with PCA (AlexNet) | ColorJitter with PCA (AlexNet) |                           |
##### Test-time Augmentations
| VGG Evaluation                       |
| ------------------------------------ |
| Resize(Q)                            |
| GridCrop(5x5)   *[Multi-Scale Eval]* |
| RandomHorizontalFlip(0.5)            |
* If the model was trained on a single scale, S, Q was chosen through {S-32, S, S+32}. If the model was trained on multiple scales, Q was chosen through {$S_{min}$, ($\frac{S_{min} + S_{max}}{2}$), $S_{max}$}. 
* In case of multi-crop evaluation, the images were cropped into a 5x5 grid (TwentyFiveCrop ?) and the evaluation results were averaged across the batch.
## Architecture

<center> Fig1: Architecture of VGG-19 (Model E in Simonyan and Zisserman, 2015), with Batch Normalization added after the Convolution Layers</center>

1. The authors reason that the effective receptive fields of two stacked `conv3x3` layers is the same as that of one `conv5x5`, and similarly three stacked `conv3x3` for one `conv7x7`, which reduces the number of parameters in the network significantly (by 81% in the case of `conv7x7`). The added benefit is the additional non-linearities between the layers, which make the decision function discriminate better, and in a sense are a nonlinear decomposition of larger kernel sizes.
2. In the VGG-C configuration, the authors also use `conv1x1` layers end of the last few blocks to increase the non-linearity of the decision function, without affecting the receptive fields of the other convolutional layers, as in the "Network in Network" architecture by Lin et al, 2014.
3. They reference the works by Ciresan et al, who had used smaller kernel sizes previously and Goodfellow et al, 2014 who had used 11 layer deep networks for street number recognition, but neither had applied it in ILSVRC. GoogLeNet is a 22 layer deep network by Szegedy et al that won ILSVRC-2014, using an ensemble of 7 models, but performs slightly worse in single model performance vs VGGNet.
## Hyperparameters

| Model Name | Parameter             | Value                                                                                                                                |
| ---------- | --------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| VGGNet     | Optimizer             | `SGD(lr = 0.01, momentum = 0.9, weight_decay = 5e-4)`                                                                                |
|            | Loss Function         | `Softmax + NLLLoss (multinomial logistic regression)`                                                                                |
|            | Batch Size            | `256 with data parallelism`                                                                                                          |
|            | LR Scheduler          | `ReduceLROnPlateau(factor = 0.1) watching val/acc`                                                                                   |
|            | Weight Initialization | Random Initialization: $W\sim\mathcal{N}(0, 0.01),\:\mathcal{b}=0$ or Xavier Initialization as described in Glorot and Bengio (2010) |
|            | Epochs                | `74`                                                                                                                                 |
* The authors trained the networks on 4x NVIDIA Titan Black GPUs with Data Parallel, which took 2-3 weeks per network, which is a speedup of 3.75x over a single GPU.
## Quantitative Evaluation
1. Using Local Response Normaliztion does not help with VGG-A, and the authors don't use it on the larger variants. However the PyTorch VGGNet implementations with Batch Normalization perform better than the non-normalized variants of the original paper.
2. Error generally decreases with increase in depth, and the use of convolutional filters with non-trivial (i.e. not 1x1) to capture spatial context performs better, given the same depth. (VGG-D performs better than VGG-C)
3. Deeper networks with smaller kernel sizes perform better than shallow networks with larger kernel sizes, confirmed by comparing VGG-B with a 5 layer deep ConvNet with 5x5 kernels, which which had an error of 7% more than VGG-B.
4. Multi Scale training significantly improves results over Single Scale, in both Dense and Multi Crop evaluations.
5. Multi Crop evaluation results in slightly better results than Dense evaluation, and using both improves the results even further, adding evidence that they are complementary.

| Model Name                          | Evaluation Method                                                                       | Top-1 Accuracy (%) | Top-5 Accuracy (%) (Val/Test) | Ops       | GFLOPs |
| ----------------------------------- | --------------------------------------------------------------------------------------- | ------------------ | ----------------------------- | --------- | ------ |
| VGG-16 (VGG-D)                      | S ~ [256, 512], Q = {256, 384, 512}, Dense and Multi Crop                               | 75.6               | 92.8 / -                      | 138e6     |        |
| VGG-19 (VGG-E)                      | S ~ [256, 512], Q = {256, 384, 512}, Dense and Multi Crop                               | 75.6               | 92.9 / -                      | 144e6     |        |
| VGG (ILSVRC-14 Submission Ensemble) | 2xC, 3xD, 2xE, Single Scale Training with S = 256 or 384 (except for 1xD), unclear eval | 75.3               | 92.5 / 92.7                   |           |        |
| VGG (Ensemble)                      | VGG-16 and VGG-19, S ~ [256, 512], Q = {256, 384, 512}, Dense and Multi Crop            | **76.3**           | **93.2 / 93.2**               |           |        |
| VGG-16 (Pytorch, Batch Norm)        |                                                                                         | 73.3               | 91.5                          | 138365992 | 15.47  |
| VGG-19 (Pytorch, Batch Norm)        |                                                                                         | 74.2               | 91.8                          | 143678248 | 19.63  |
