#2D Image & Filter
image_grey = np.array([
    [1, 1, 0],
    [0, 1, 0],
    [1, 0, 1]
])

filter_grey = np.array([
    [1, 0],
    [0, 1]
])



#2D Convolution
def apply_convolution(image_grey, filter_grey):
    size = filter_grey.shape[0]
    height, width = image_grey.shape
    result = np.zeros((height-size+1, width-size+1))
    
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            result[i, j] = np.sum(image_grey[i:i+size, j:j+size]*filter_grey)

    return result

feature_map = apply_convolution(image_grey, filter_grey)



#Conv2d
conv_layer_stride = nn.Conv2d(1, 1, kernel_size=3, stride=1)
output_tensor_stride = conv_layer_stride(input_tensor)

conv_layer_stride_padding = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
output_tensor_stride_padding = conv_layer_stride_padding(input_tensor)


#RELU
import torch.nn.functional as F

conv_layer_stride = nn.Conv2d(1, 1, kernel_size=3, stride=1)
output_tensor_stride = conv_layer_stride(input_tensor)

mean_before_relu = output_tensor_stride.mean().item()
output_tensor_stride_relu = F.relu(output_tensor_stride)
mean_after_relu = output_tensor_stride_relu.mean().item()



#Pooling
max_pool_layer = nn.MaxPool2d(kernel_size=2, stride=2)
pooled_feature_map = max_pool_layer(feature_map)

avg_pool_layer = nn.AvgPool2d(kernel_size=2, stride=2)
pooled_feature_map = avg_pool_layer(feature_map)



#Fully Connected Layer
fc_layer = nn.Linear(in_features=75, out_features=3)
activation = nn.ReLU()

output = activation(fc_layer(x_flattened))


fc_layer1 = nn.Linear(in_features=75, out_features=20)
fc_layer2 = nn.Linear(in_features=20, out_features=3)
activation = nn.ReLU()  

output1 = activation(fc_layer1(x_flattened))
output2 = fc_layer2(output1)  



#Softmax Function
fc_layer1 = nn.Linear(in_features=75, out_features=40)
fc_layer2 = nn.Linear(in_features=40, out_features=20)
fc_layer3 = nn.Linear(in_features=20, out_features=3)
activation = nn.ReLU()

output1 = activation(fc_layer1(x_flattened))
output2 = activation(fc_layer2(output1))
output3 = fc_layer3(output2)

probabilities = F.softmax(output3, dim=1)