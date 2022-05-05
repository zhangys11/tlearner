# About

We provide a set of transfer learning-based model training and evaluation functions.  
The series of Efficient Net models are supported. 
There is always a trade-off between model size and accuracy. Our guideline is as follows:     
For tfjs apps, use EfficientNetB0 or EfficientNetB1; For tf-lite apps, use EfficientNetB2 ~ B4; For desktop apps, use EfficientNetB5 and above. 

The following table is from keras website:
<table>
<thead>
<tr>
<th>Model</th>
<th>Size (MB)</th>
<th>Top-1 Accuracy</th>
<th>Top-5 Accuracy</th>
<th>Parameters</th>
<th>Depth</th>
<th>Time (ms) per inference step (CPU)</th>
<th>Time (ms) per inference step (GPU)</th>
</tr>
</thead>
<tbody>
<tr>
<td><a href="xception">Xception</a></td>
<td>88</td>
<td>79.0%</td>
<td>94.5%</td>
<td>22.9M</td>
<td>81</td>
<td>109.4</td>
<td>8.1</td>
</tr>
<tr>
<td><a href="vgg/#vgg16-function">VGG16</a></td>
<td>528</td>
<td>71.3%</td>
<td>90.1%</td>
<td>138.4M</td>
<td>16</td>
<td>69.5</td>
<td>4.2</td>
</tr>
<tr>
<td><a href="vgg/#vgg19-function">VGG19</a></td>
<td>549</td>
<td>71.3%</td>
<td>90.0%</td>
<td>143.7M</td>
<td>19</td>
<td>84.8</td>
<td>4.4</td>
</tr>
<tr>
<td><a href="resnet/#resnet50-function">ResNet50</a></td>
<td>98</td>
<td>74.9%</td>
<td>92.1%</td>
<td>25.6M</td>
<td>107</td>
<td>58.2</td>
<td>4.6</td>
</tr>
<tr>
<td><a href="resnet/#resnet50v2-function">ResNet50V2</a></td>
<td>98</td>
<td>76.0%</td>
<td>93.0%</td>
<td>25.6M</td>
<td>103</td>
<td>45.6</td>
<td>4.4</td>
</tr>
<tr>
<td><a href="resnet/#resnet101-function">ResNet101</a></td>
<td>171</td>
<td>76.4%</td>
<td>92.8%</td>
<td>44.7M</td>
<td>209</td>
<td>89.6</td>
<td>5.2</td>
</tr>
<tr>
<td><a href="resnet/#resnet101v2-function">ResNet101V2</a></td>
<td>171</td>
<td>77.2%</td>
<td>93.8%</td>
<td>44.7M</td>
<td>205</td>
<td>72.7</td>
<td>5.4</td>
</tr>
<tr>
<td><a href="resnet/#resnet152-function">ResNet152</a></td>
<td>232</td>
<td>76.6%</td>
<td>93.1%</td>
<td>60.4M</td>
<td>311</td>
<td>127.4</td>
<td>6.5</td>
</tr>
<tr>
<td><a href="resnet/#resnet152v2-function">ResNet152V2</a></td>
<td>232</td>
<td>78.0%</td>
<td>94.2%</td>
<td>60.4M</td>
<td>307</td>
<td>107.5</td>
<td>6.6</td>
</tr>
<tr>
<td><a href="inceptionv3">InceptionV3</a></td>
<td>92</td>
<td>77.9%</td>
<td>93.7%</td>
<td>23.9M</td>
<td>189</td>
<td>42.2</td>
<td>6.9</td>
</tr>
<tr>
<td><a href="inceptionresnetv2">InceptionResNetV2</a></td>
<td>215</td>
<td>80.3%</td>
<td>95.3%</td>
<td>55.9M</td>
<td>449</td>
<td>130.2</td>
<td>10.0</td>
</tr>
<tr>
<td><a href="mobilenet">MobileNet</a></td>
<td>16</td>
<td>70.4%</td>
<td>89.5%</td>
<td>4.3M</td>
<td>55</td>
<td>22.6</td>
<td>3.4</td>
</tr>
<tr>
<td><a href="mobilenet/#mobilenetv2-function">MobileNetV2</a></td>
<td>14</td>
<td>71.3%</td>
<td>90.1%</td>
<td>3.5M</td>
<td>105</td>
<td>25.9</td>
<td>3.8</td>
</tr>
<tr>
<td><a href="densenet/#densenet121-function">DenseNet121</a></td>
<td>33</td>
<td>75.0%</td>
<td>92.3%</td>
<td>8.1M</td>
<td>242</td>
<td>77.1</td>
<td>5.4</td>
</tr>
<tr>
<td><a href="densenet/#densenet169-function">DenseNet169</a></td>
<td>57</td>
<td>76.2%</td>
<td>93.2%</td>
<td>14.3M</td>
<td>338</td>
<td>96.4</td>
<td>6.3</td>
</tr>
<tr>
<td><a href="densenet/#densenet201-function">DenseNet201</a></td>
<td>80</td>
<td>77.3%</td>
<td>93.6%</td>
<td>20.2M</td>
<td>402</td>
<td>127.2</td>
<td>6.7</td>
</tr>
<tr>
<td><a href="nasnet/#nasnetmobile-function">NASNetMobile</a></td>
<td>23</td>
<td>74.4%</td>
<td>91.9%</td>
<td>5.3M</td>
<td>389</td>
<td>27.0</td>
<td>6.7</td>
</tr>
<tr>
<td><a href="nasnet/#nasnetlarge-function">NASNetLarge</a></td>
<td>343</td>
<td>82.5%</td>
<td>96.0%</td>
<td>88.9M</td>
<td>533</td>
<td>344.5</td>
<td>20.0</td>
</tr>
<tr>
<td><a href="efficientnet/#efficientnetb0-function">EfficientNetB0</a></td>
<td>29</td>
<td>77.1%</td>
<td>93.3%</td>
<td>5.3M</td>
<td>132</td>
<td>46.0</td>
<td>4.9</td>
</tr>
<tr>
<td><a href="efficientnet/#efficientnetb1-function">EfficientNetB1</a></td>
<td>31</td>
<td>79.1%</td>
<td>94.4%</td>
<td>7.9M</td>
<td>186</td>
<td>60.2</td>
<td>5.6</td>
</tr>
<tr>
<td><a href="efficientnet/#efficientnetb2-function">EfficientNetB2</a></td>
<td>36</td>
<td>80.1%</td>
<td>94.9%</td>
<td>9.2M</td>
<td>186</td>
<td>80.8</td>
<td>6.5</td>
</tr>
<tr>
<td><a href="efficientnet/#efficientnetb3-function">EfficientNetB3</a></td>
<td>48</td>
<td>81.6%</td>
<td>95.7%</td>
<td>12.3M</td>
<td>210</td>
<td>140.0</td>
<td>8.8</td>
</tr>
<tr>
<td><a href="efficientnet/#efficientnetb4-function">EfficientNetB4</a></td>
<td>75</td>
<td>82.9%</td>
<td>96.4%</td>
<td>19.5M</td>
<td>258</td>
<td>308.3</td>
<td>15.1</td>
</tr>
<tr>
<td><a href="efficientnet/#efficientnetb5-function">EfficientNetB5</a></td>
<td>118</td>
<td>83.6%</td>
<td>96.7%</td>
<td>30.6M</td>
<td>312</td>
<td>579.2</td>
<td>25.3</td>
</tr>
<tr>
<td><a href="efficientnet/#efficientnetb6-function">EfficientNetB6</a></td>
<td>166</td>
<td>84.0%</td>
<td>96.8%</td>
<td>43.3M</td>
<td>360</td>
<td>958.1</td>
<td>40.4</td>
</tr>
<tr>
<td><a href="efficientnet/#efficientnetb7-function">EfficientNetB7</a></td>
<td>256</td>
<td>84.3%</td>
<td>97.0%</td>
<td>66.7M</td>
<td>438</td>
<td>1578.9</td>
<td>61.6</td>
</tr>
<tr>
<td><a href="efficientnet_v2/#efficientnetv2b0-function">EfficientNetV2B0</a></td>
<td>29</td>
<td>78.7%</td>
<td>94.3%</td>
<td>7.2M</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td><a href="efficientnet_v2/#efficientnetv2b1-function">EfficientNetV2B1</a></td>
<td>34</td>
<td>79.8%</td>
<td>95.0%</td>
<td>8.2M</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td><a href="efficientnet_v2/#efficientnetv2b2-function">EfficientNetV2B2</a></td>
<td>42</td>
<td>80.5%</td>
<td>95.1%</td>
<td>10.2M</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td><a href="efficientnet_v2/#efficientnetv2b3-function">EfficientNetV2B3</a></td>
<td>59</td>
<td>82.0%</td>
<td>95.8%</td>
<td>14.5M</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td><a href="efficientnet_v2/#efficientnetv2s-function">EfficientNetV2S</a></td>
<td>88</td>
<td>83.9%</td>
<td>96.7%</td>
<td>21.6M</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td><a href="efficientnet_v2/#efficientnetv2m-function">EfficientNetV2M</a></td>
<td>220</td>
<td>85.3%</td>
<td>97.4%</td>
<td>54.4M</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td><a href="efficientnet_v2/#efficientnetv2l-function">EfficientNetV2L</a></td>
<td>479</td>
<td>85.7%</td>
<td>97.5%</td>
<td>119.0M</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
</tbody>
</table>

# How to use

from tlearner.efficientnet import transfer_learner
learner = transfer_learner("flower_customEfficientNetB0_model", W = 224)

# Jupyter notebooks

Under /notebooks, we provide two examples. One is flower image classification; the other is fundus image classification.

# Deployment

After training, you will get a keras h5 model file. You can further convert it to tflite format, or tfjs format (efficient net is not supported yet).  
Then you can deploy on mobile device or browser-based apps.