import cv2
import os

cam = cv2.VideoCapture(0)
detector=cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')

subjects = []
with open('subjects.txt', 'r') as filehandle:  
    for line in filehandle:
        currentPlace = line[:-1]
        subjects.append(currentPlace)
    
Name=raw_input('Enter your name: ');
Id=str(len(subjects));
sampleNum=0
os.mkdir("training-data/s"+Id)
while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        sampleNum=sampleNum+1
        cv2.imwrite("training-data/s"+Id+"/User."+Id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
        cv2.imshow('frame',img)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    elif sampleNum>20:
        break
    
subjects.append(Name);
with open('subjects.txt', 'w') as filehandle:  
    for listitem in subjects:
        filehandle.write('%s\n' % listitem)
cam.release()
cv2.destroyAllWindows()
