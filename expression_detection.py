import cv2
from deepface import DeepFace

video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    exit()
    
while True:
    result, frame = video_capture.read()
    
    if not result:
        break
    
    result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
    
    emotion = result[0]['dominant_emotion']
    
    cv2.putText(frame, f'Emotion: {emotion}', (50, 50), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow('Expression Detection', frame)
    
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()