import cv2
import dlib
import numpy as np
import os
from tqdm import tqdm

# 얼굴 이미지를 로드하고 인식 방법을 학습하는 함수
def load_known_faces(known_faces_dir):
    known_faces = []
    known_names = []
    for name in tqdm(os.listdir(known_faces_dir), desc="Processing faces", unit="person"):
        for filename in os.listdir(f"{known_faces_dir}/{name}"):
            image = cv2.imread(f"{known_faces_dir}/{name}/{filename}")
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_locations = detector(rgb_image, 1)
            for face_location in face_locations:
                shape = predictor(rgb_image, face_location)
                face_encoding = np.array(facerec.compute_face_descriptor(rgb_image, shape))
                known_faces.append(face_encoding)
                known_names.append(name)
    return known_faces, known_names

# dlib의 얼굴 탐지기(HOG 기반) 및 형태 예측기를 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('face_duldul/shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('face_duldul/dlib_face_recognition_resnet_model_v1.dat')

def main():
    known_faces, known_names = load_known_faces("face_data")

    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = detector(rgb_frame, 1)
        face_encodings = [np.array(facerec.compute_face_descriptor(rgb_frame, predictor(rgb_frame, face_location))) for face_location in face_locations]

        face_names = []
        for face_encoding in face_encodings:
            matches = np.linalg.norm(known_faces - face_encoding, axis=1)
            best_match_index = np.argmin(matches)
            if matches[best_match_index] < 0.6:
                name = known_names[best_match_index]
            else:
                name = "Unknown"
            face_names.append(name)

        for (face_location, name) in zip(face_locations, face_names):
            top, right, bottom, left = (face_location.top(), face_location.right(), face_location.bottom(), face_location.left())
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()                                                                                                   