import base64
import requests
import json
import cv2

image_path = "image3.jpg"
url = "http://localhost:4000/sam"

param = {
  "image": str(base64.b64encode(open(image_path, 'rb').read()), 'utf-8'),
  "iou_thresh": 0.9,
  "stability_thresh": 0.8,
  "clip_thresh": 0.96,
  "query": "dog,horse"
}

req = requests.post(url, data=json.dumps(param))

data = json.loads(req.text)

image = cv2.imread(image_path)
for box in data['bbox']:
    p1, p2 = (box['x1'], box['y1']), (box['x2'], box['y2'])
    cv2.rectangle(image, p1, p2, (0, 0, 255), 1)

cv2.imwrite("results.jpg", image)