import cv2
from scipy.cluster import vq
print "Enter the input file name"
input_filename = raw_input()
print "Enter the output file name"
output_filename = raw_input()
# Reading the image in BGR format.
img = cv2.imread(input_filename)
#Reshaping the image matrix
z = img.reshape((-1,3))
#Getting the clusters using k-means clusttering algorithm
k = 2
center,dist = vq.kmeans(z,k)
code,distance = vq.vq(z,center)
res = center[code]
res2 = res.reshape((img.shape))
cv2.imwrite('temp_img.jpg',res2)
img1 = cv2.imread('temp_img.jpg')
z = img1.reshape((-1,3))
k = 2           # Number of clusters
center,dist = vq.kmeans(z,k)
 
 
center[0] = [255,255,255]
center[1] = [100, 100, 100]
code,distance = vq.vq(z,center)
res = center[code]
res2 = res.reshape((img1.shape))
cv2.imwrite('temp_img.jpg',res2)
 
#Second round of quantization
temp_img_2 = cv2.imread('temp_img.jpg')
z = temp_img_2.reshape((-1,3))
k = 2           # Number of clusters
center,dist = vq.kmeans(z,k)
center[0] = [255,255,255]
center[1] = [50, 50, 50]
code,distance = vq.vq(z,center)
res = center[code]
res2 = res.reshape((temp_img_2.shape))
cv2.imwrite('temp_img.jpg',res2)
 
#Third round of quantization.
temp_img_3 = cv2.imread('temp_img.jpg')
z = temp_img_3.reshape((-1,3))
k = 2           # Number of clusters
center,dist = vq.kmeans(z,k)
center[0] = [255,255,255]
center[1] = [10, 10, 10]
code,distance = vq.vq(z,center)
res = center[code]
res2 = res.reshape((temp_img_3.shape))
cv2.imwrite(output_filename,res2)
