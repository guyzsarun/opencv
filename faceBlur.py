import cv2
import numpy as np
cap = cv2.VideoCapture(0)


face_cascade = cv2.CascadeClassifier("./haarcascades/haarcascade_frontalface_default.xml")

def detect_face(img):
    
    face_rects = face_cascade.detectMultiScale(img) 
    #face_rects = face_cascade.detectMultiScale(img,scaleFactor=1.3, minNeighbors=5) 
    img_copy=img.copy()
    
    mask = np.full((img.shape[:2]+(1,)), 0, dtype=np.uint8)

    
    blur_img=cv2.GaussianBlur(img_copy,(51,51),10)
    
    #Mask face
    for (x,y,w,h) in face_rects: 
        cv2.circle(mask,(int((x + x + w)/2), int((y + y + h)/2)),int(h/2),(255),-1)
        
    mask_inv = cv2.bitwise_not(mask)
    bg = cv2.bitwise_and(img, img, mask=mask_inv)
    face = cv2.bitwise_and(blur_img, blur_img, mask=mask)
    
    #Combined img
    combined=cv2.add(face,bg)
    
    return combined
    

while True:
    ret, frame = cap.read()
    frame=cv2.flip(frame,1)
    
    blur_face=detect_face(frame)
    
    cv2.imshow('frame',blur_face)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()