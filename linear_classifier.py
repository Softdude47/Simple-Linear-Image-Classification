import cv2
import numpy as np

# initialize class labels and genrator
labels = ["dog", "cat", "panda"]
np.random.seed(1)

# initialize weight matrix and bias
# this random matrix is a worked example though
W = np.random.randn(3, 3072)
b = np.random.randn(3)

# load image and flatten image
image = cv2.imread("dog2.jpg")
X = cv2.resize(image, (32, 32)).flatten()

# calculate the dot product between weight matrix
# and the image before adding bias
scores = W.dot(X) + b

# display each classes and their associated scores
for (label, score) in zip(labels, scores):
    print(f"[INFO]: {label} : {score}%")

# getting the index of the predicted class i.e
# the class with highest score
y_pred = np.argmax(scores)

# display the image with the predicted class label on it
cv2.putText(image, labels[y_pred], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
cv2.imshow(f"Label: {labels[y_pred]}", image)
cv2.waitKey(0) & 0xff
cv2.destroyAllWindows()