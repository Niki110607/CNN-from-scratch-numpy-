# This is a guide to create the structure for the model
Since there are only three layers it is pretty simple.

## Convolution Layer
[layer_type="conv", input_channels=int, filter_amount(output_channels)=int, filter_size=int(3,5,7)]

## Pooling Layer
[layer_type="pool", size=int(2)]

## Fully Connected Layer
[layer="FC", input_size=int, output_size=int(layer_size), softmax=bool(only True if this is the last layer]


## Final structure
To add those layers into a final model you have to create a list of those layers.
This is the structure i used in my digit model.

structure = [  
["conv", 1, 4, 3],  
["pool", 2],  
["conv", 4, 8, 3],  
["FC", 1568, 512, False],  
["FC", 512, 256, False],  
["FC", 256, 10, True]
]
