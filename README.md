# AI Body Tracking

Experiments with different pose estimation libraries and models.

## Files

- `mediapipe_clean.py` - MediaPipe holistic tracking (face, pose, hands)
- `ai.py` - Basic MediaPipe webcam feed with pose detection
- `opencv_pose.py` - OpenCV DNN pose estimation using MobileNet
- `posenet_demo.py` - TensorFlow Hub MoveNet pose detection
- `openpose_pose_mpi.prototxt` - OpenPose model configuration

## Requirements

```bash
pip install mediapipe opencv-python tensorflow tensorflow-hub
```

## Usage

Run any script to start webcam pose tracking:
```bash
python mediapipe_clean.py
python posenet_demo.py
```
