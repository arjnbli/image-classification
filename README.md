# Deep Residual Network Implementation for Tiny ImageNet

This is a tensorflow implementation of a 14 layer deep residual network based on the ResNet-18 architecture.
The last 4 layers have been removed because of the smaller size of images in Tiny ImageNet (64x64x3) as compared
to (256x256x3) in ImageNet.

Since the test labels are not provided, the validation set with 10,000 examples is used as the test set and 10,000 
examples are taken from the training set to the create the test set. This gives us a training set of 90,000 examples,
a validation set of 10,000 examples and a test set of 10,000 examples.

There are 200 distinct output labels. However, there are only 500 training examples per output label in the training
set. As a consequence, image augementation is used. The image preprocessing steps involve - random flips left and right,
random hue and random crop. The random crop reduces the image size from (64x64x3) to (56x56x3). 

The cropped image is used as input to a convolutional layer with 64 (7x7) filters. This is followed by three residual
residual blocks with each block comprising 4 convolutional layers. The first residual block has 64 (3x3) filters, 
the second residual block has 128 (3x3) filters and the third residual block has 256 (3x3) filters. This transforms 
the pre-processed input image of shape (56x56x3) to (7x7x256). This is followed by global average pooling which further
reduces the image to (7x7x1). The network ends in a fully connected layer which produces a vector of shape (200x1).

In order to match dimensions for residual connections, (1x1) convolutions with stride 2 are performed.

The categorical cross entropy function is used as the optimization objective. The optimization is performed using the
Adam optimizer with weight decay.

Early stopping is used to stop training if no improvement in validation set accuracy occurs for a number of iterations.
