import cv2
cap = cv2.VideoCapture(0)


face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')

def detect_face(img):
    
    face_rects = face_cascade.detectMultiScale(img,scaleFactor=1.2, minNeighbors=5) 
    img_copy=img.copy()
    for (x,y,w,h) in face_rects: 
        face=img[y:y+h,x:x+w]
        face=cv2.GaussianBlur(face,(51,51),10)
        img[y:y+h,x:x+w]=face
        
    return img
    

while True:
    
    ret, frame = cap.read()
    frame=cv2.flip(frame,1)
    face=detect_face(frame)
    
    cv2.imshow('frame',face)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()