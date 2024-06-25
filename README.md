# ImageSegmentationCustomNetwork

An encoder based image segmentation model. This is experimental. The purpose is not to get state of the art performance so much as to test out a custom model architecture that I'm working on.

I use a mixture of experts based architecture to simulate 3d convolution layers using 3 2d layers in the (depth, height), (depth, width), and (height, width) directions.

My custom architecture is compared to a model based on the same architecture using all 2d layers, and in the future, will be compared to an architecture using all traditional 3d convolution layers. 
