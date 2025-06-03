# Football Analysis AI ⚽📊

This project uses computer vision and deep learning to analyze football match videos. It includes player and ball detection, tracking, team assignment, and ball possession analysis using YOLO and custom logic.

---

## 📦 Features

- Detect players, referees, goalkeepers, and the ball using YOLO
- Track objects using ByteTrack
- Assign teams using KMeans clustering on jersey colors
- Interpolate missing ball positions
- Detect which player has the ball
- Calculate team ball possession over time
- Draw annotations on each frame and generate output video

---

## 🛠 Requirements

- Python 3.x
- The following Python libraries:
  - ultralytics
  - supervision
  - opencv-python
  - numpy
  - matplotlib
  - pandas
 
---

## 🔗 Sample video link

https://www.kaggle.com/datasets/saberghaderi/-dfl-bundesliga-460-mp4-videos-in-30sec-csv?resource=download

Install all dependencies using:

```bash
pip install -r requirements.txt
