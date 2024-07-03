# face_duldul

## Face Recognition with OpenCV and Dlib
### Welcome to the Face Recognition project! This project demonstrates how to perform face recognition using OpenCV and Dlib in Python. The code will load known face images, encode them, and then match faces from a live video feed against these known encodings.

Features
Load and Cache Known Faces: Load known face encodings from images and cache them for faster future processing.
Real-Time Face Recognition: Detect and encode faces in real-time using your webcam.
Face Identification: Identify and label recognized faces in the video feed.
Parallel Processing: Use parallel processing for faster face encoding.
Requirements
Python 3.x
OpenCV
Dlib
Numpy
Tqdm
Installation
Follow these steps to set up the project on your local machine:

Clone the repository:

bash
코드 복사
git clone https://github.com/yourusername/face-recognition.git
cd face-recognition
Install the required Python packages:

bash
코드 복사
pip install -r requirements.txt
Download the required Dlib models and place them in the face_duldul directory:

shape_predictor_68_face_landmarks.dat
dlib_face_recognition_resnet_model_v1.dat
Extract the downloaded .dat.bz2 files:

bash
코드 복사
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d dlib_face_recognition_resnet_model_v1.dat.bz2
Usage
Prepare Known Face Images:

Place known face images in the face_data directory.
Organize images in subdirectories named after the people in the images. For example:
코드 복사
face_data/
├── person1/
│   ├── image1.jpg
│   └── image2.jpg
└── person2/
    ├── image1.jpg
    └── image2.jpg
Run the Main Script:

Start the face recognition from your webcam:
bash
코드 복사
python main.py
Code Overview
load_known_faces
python
코드 복사
def load_known_faces(known_faces_dir, cache_file="face_encodings_cache.pkl"):
    # Load or compute face encodings
Description: Loads and encodes known faces from the specified directory. If a cache file exists, it loads the encodings from there to save time.
encode_faces_in_frame
python
코드 복사
def encode_faces_in_frame(rgb_frame, face_locations):
    # Encode faces in a single frame using parallel processing
Description: Encodes faces found in a single frame using parallel processing.
main
python
코드 복사
def main():
    # Main function to capture video and recognize faces
Description: Captures video from the webcam, detects faces, matches them against known encodings, and displays the results in real-time.
Acknowledgments
This project uses the following open-source libraries:

OpenCV
Dlib
Tqdm
License
This project is licensed under the MIT License. See the LICENSE file for details.
