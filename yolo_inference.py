from ultralytics import YOLO

model = YOLO("models/best.pt")

results = model.predict('input-videos/sample1.mp4',save=True)
print(results[0])
print('----------------------------')
for box in results[0].boxes:
    print(box)