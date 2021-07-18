import cv2
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
img = cv2.imread('face.jpg')
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
cv2.rectangle(img, (139, 39), (139+112, 39+112), (0, 255, 0), 2)
print(face_coordinates)
cv2.imshow("Clever Programmer Face Detector", img)
cv2.waitKey()