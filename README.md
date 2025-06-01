 Real-Time Detection and Tracking of Specific Objects in Videos

 
This project implements a real-time computer vision system that detects and tracks specific objects—such as ambulances, fire trucks, or police cars—within a video stream. It combines YOLOv8 for object detection with DeepSORT for tracking, and features an interactive Streamlit web interface for video upload, object selection, and real-time visualization.


 Project Objective
Develop a real-time system to:

Detect user-specified objects in videos.

Track their movement with consistent IDs across frames.

Log positions and timestamps.

Display alerts when target objects appear or disappear.

 Features
 YOLOv8 detection: Efficient and accurate real-time object detection.

 DeepSORT tracking: Robust identity tracking across video frames.

Bounding boxes with labels: Displayed directly on video frames.

CSV logging: Timestamp, object ID, class, and coordinates saved.

Alert system: UI notification when searched object is detected.

Streamlit web app: Upload video, select object, visualize results.

Performance evaluation: mAP, precision, recall, ID switches.

Handles absence of object: Visual + textual cue if object is not present.

Tech Stack
Python

YOLOv8 (Ultralytics)

DeepSORT

OpenCV

Streamlit

Pandas / NumPy / Matplotlib

Project Structure
bash
CopierModifier
├── app/
│   └── app.py               # Streamlit frontend
├── models/
│   └── yolov8_weights.pt    # YOLOv8 pretrained weights
├── utils/
│   └── yolo_video.py        # Detection and tracking logic
├── main.py                  # CLI-based video processing
├── logs/
│   └── object_log.csv       # Auto-generated detection log
├── requirements.txt
├── README.md
└── LICENSE
Installation
bash
CopierModifier
# Clone the repository
git clone https://github.com/Pontiane21K/Object_detection_Group9.git
cd real-time-object-tracking

# Create and activate virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
Usage
Option 1: Run Streamlit App
bash
CopierModifier
streamlit run app/app.py
Upload a video (.mp4, .avi, etc.)

Select the object class (e.g., ambulance, car, etc.)

Watch real-time detection & tracking with alerts

Download CSV log file of detections

Option 2: CLI (Command-Line)
bash
CopierModifier
python main.py --video "path/to/video.mp4" --target_object "ambulance"
Output
Bounding Boxes: Tracked objects with class and ID

CSV Log File: logs/object_log.csv
Columns: timestamp, object_id, class_name, x1, y1, x2, y2

Evaluation Metrics
Metric	Description
Precision	Accuracy of positive detections
Recall	Fraction of actual positives detected
mAP	Mean average precision over classes
ID Switches	Number of times tracked ID was changed

Screenshots
Add screenshots or GIFs here showing your app, detection in action, and output logs.

Challenges & Improvements
Challenges
Handling multiple similar objects (e.g., cars vs. ambulances)

Dealing with occlusion and motion blur

Distinguishing between similar classes

Future Improvements
Add audio-based siren detection for false positive filtering.

Predict object direction and speed.

Generate a heatmap of object appearance zones.

Web Demo
A live demo is available at: https://objectdetectiongroup9-web-app.streamlit.app/

License
This project is licensed under the MIT License.

Contributors
Your Name - Developer, AIMS Senegal

Contact
For any queries or contributions, contact kouecking.p.blondele@aims-senegal.org or open an issue.
