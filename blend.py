import cv2

image1 = cv2.imread('image/dino.jpg')
image2 = cv2.imread('image/dino2.jpg')

resize = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

blended = cv2.addWeighted(image1, 0.5, resize, 0.5, 0)

cv2.imwrite('blended_dino.jpg', blended)
cv2.imshow('blened_dino', blended)

cv2.waitKey(0)
cv2.destroyAllWindows()