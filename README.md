# Face Recognition with OpenCV and Dlib with Jaegal-DulDul!

Welcome to the Face Recognition project! This project demonstrates how to perform face recognition using OpenCV and Dlib in Python. The code will load known face images, encode them, and then match faces from a live video feed against these known encodings.

## Features

- **Load and Cache Known Faces:** Load known face encodings from images and cache them for faster future processing.
- **Real-Time Face Recognition:** Detect and encode faces in real-time using your webcam.
- **Face Identification:** Identify and label recognized faces in the video feed.
- **Parallel Processing:** Use parallel processing for faster face encoding.

## Requirements

- Python 3.x
- OpenCV
- Dlib
- Numpy
- Tqdm

## Installation

Follow these steps to set up the project on your local machine:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/face-recognition.git
    cd face-recognition
    ```

2. **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Download the required Dlib models and place them in the `face_duldul` directory:**
    - [shape_predictor_68_face_landmarks.dat](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
    - [dlib_face_recognition_resnet_model_v1.dat](http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2)

4. **Extract the downloaded `.dat.bz2` files:**
    ```bash
    bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
    bzip2 -d dlib_face_recognition_resnet_model_v1.dat.bz2
    ```

## Usage

1. **Prepare Known Face Images:**
    - Place known face images in the `face_data` directory.
    - Organize images in subdirectories named after the people in the images. For example:
    ```
    face_data/
    ├── person1/
    │   ├── image1.jpg
    │   └── image2.jpg
    └── person2/
        ├── image1.jpg
        └── image2.jpg
    ```

2. **Run the Main Script:**
    - Start the face recognition from your webcam:
    ```bash
    python main.py
    ```

## Code Overview

### load_known_faces

```python
def load_known_faces(known_faces_dir, cache_file="face_encodings_cache.pkl"):
    # Load or compute face encodings
