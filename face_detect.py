import cv2 as cv

img = cv.imread('images/group 2.jpg')
cv.imshow('Lady',img)

#gray
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('Gray',gray)
#loading classifier
haar_cascade = cv.CascadeClassifier('haar_face.xml')
#getting detected rectangle coordinates
face_rect = haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=6)

print(f'Number of faces detected= {len(face_rect)}')

#drawing rectangle around faces
for (x,y,w,h) in face_rect:
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

cv.imshow('Detected faces', img)
cv.waitKey(0)