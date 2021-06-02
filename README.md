# Embedding
# Image Embedding with Deep Neural Network Architectures

**Steps through run the code**

1. PATH is the path where this .ipynb should be placed. In this code PATH = '/content/drive/MyDrive/Colab Notebooks'
2. There should be folder named '/embedding-logs' in the PATH folder that would contain the log files for embedding.
3. Xception Model [1] has been used to extract feature and classification.
4. Bangla Handwritten Nurerals Dataset has been used in this example. But any other image dataset of which the labels are separated by different folders can be visualized with this script. 
5. The image dimension is (3, 28, 28). 
6. After run the TensorBoard, select "Projector" option to visualize the embedding. 

# 3D Visualization of Embedding
![3D Visualization](https://user-images.githubusercontent.com/9729244/120443913-ef33bf00-c3a8-11eb-996f-256aec4762e7.PNG)


# 2D Visualization of Embedding
![2D Visualization](https://user-images.githubusercontent.com/9729244/120444041-0f637e00-c3a9-11eb-9259-f1c16c5f96ce.PNG)




# Reference

[1] Chollet, François. "Xception: Deep learning with depthwise separable convolutions." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.

