import face_recognition
import os

known = face_recognition.load_image_file('image/ronaldo.jpg')
known_emb = face_recognition.face_encodings(known)[0]


folder_path = 'face'
for img in os.listdir(folder_path):
    face_path = f'{folder_path}/{img}'
    face = face_recognition.load_image_file(face_path)
    face_emb = face_recognition.face_encodings(face)[0]

    res = face_recognition.compare_faces([known_emb], face_emb)

    print(res[0])



