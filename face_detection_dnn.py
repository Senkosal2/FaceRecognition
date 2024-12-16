import cv2
from deepface import DeepFace

def detect_face(image):
    net = cv2.dnn.readNetFromTensorflow('opencv_face_detector_uint8.pb', 'opencv_face_detector.pbtxt')

    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
    net.setInput(blob)
    
    face = net.forward()
    
    for i in range(face.shape[2]):
        confidence = face[0, 0, i, 2]
        if confidence > 0.5:
            x1 = int(face[0, 0, i, 3] * image.shape[1])
            y1 = int(face[0, 0, i, 4] * image.shape[0])
            x2 = int(face[0, 0, i, 5] * image.shape[1])
            y2 = int(face[0, 0, i, 6] * image.shape[0])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            result = DeepFace.analyze(image, actions=['emotion'], enforce_detection=False)
    
            cv2.putText(image, result[0]['dominant_emotion'], (x1, y1-10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            
            print('x1: ', x1 , ' x2: ', x2)
            print('y1: ', y1 , ' y2: ', y2)
    #return face


# capture video in real-time
#video_capture = cv2.VideoCapture('http://192.168.1.56:8080/video')
video_capture = cv2.VideoCapture(0)

# main process
while True:
    
    result, frame = video_capture.read()
    
    if not result:
        break
    
    detect_face(frame)
    #detect_expression(frame)
    
    cv2.imshow('Camera', frame)
    
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
    
video_capture.release()
cv2.destroyAllWindows()
    