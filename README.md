# Deep Residual Network Implementation for Tiny ImageNet

This is a tensorflow implementation of a 14 layer deep residual network based on the ResNet-18 architecture
The last 4 layers have been removed because of the smaller size of images in Tiny ImageNet (64x64x3) as compared
to (256x256x3) in ImageNet.

Since the test labels are not provided, the validation set with 10,000 examples is used as the test set and 10,000 
images and corresponding labels are taken from the training set to the create the test set. This leaves us with a 
training set consisting of 90,000 examples.

There are 200 distinct output labels. However, there are only 500 training examples per output label. As a consequence,
image augementation is used. The image preprocessing steps involve - random flips left and right, random hue and 
random crop. A random crop reduces the image size from (64x64x3) to (56x56x3).

In order to match dimensions for residual connections, 1x1 convolutions  with stride 2 are performed.

The categorical cross function is used as the optimization objective. The optimization is performed using the Adam
optimizer with weight decay.

Early stopping is used to stop training if no imrovement in accuracy occurs for a number of iterations.

