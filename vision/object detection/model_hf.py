import cv2
import requests
from PIL import Image
from transformers import pipeline
# from io import BytesIO

# Download an image with cute cats
url = "/Users/kaushiktummalapalli/Desktop/Image Processing/Hack-Princeton/vision/captioning/p.jpeg"
# image_data = requests.get(url, stream=True).raw
image = Image.open(url)

# Allocate a pipeline for object detection
object_detector = pipeline('object-detection')
# print(object_detector(image))
helper = object_detector(image)


# Load the image
img = cv2.imread(
    "/Users/kaushiktummalapalli/Desktop/Image Processing/Hack-Princeton/vision/captioning/p.jpeg")

# Loop through each object and draw the corresponding bounding box
for obj in helper:
    xmin = obj['box']['xmin']
    ymin = obj['box']['ymin']
    xmax = obj['box']['xmax']
    ymax = obj['box']['ymax']
    label = obj['label']

    # Draw the bounding box on the image
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    # Add the label text to the image
    cv2.putText(img, label, (xmin, ymin - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Display the final image
# cv2.imshow("Image", img)
cv2.imwrite('test.jpg', img)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

"""
[{'score': 0.9982201457023621,
  'label': 'remote',
  'box': {'xmin': 40, 'ymin': 70, 'xmax': 175, 'ymax': 117}},
 {'score': 0.9960021376609802,
  'label': 'remote',
  'box': {'xmin': 333, 'ymin': 72, 'xmax': 368, 'ymax': 187}},
 {'score': 0.9954745173454285,
  'label': 'couch',
  'box': {'xmin': 0, 'ymin': 1, 'xmax': 639, 'ymax': 473}},
 {'score': 0.9988006353378296,
  'label': 'cat',
  'box': {'xmin': 13, 'ymin': 52, 'xmax': 314, 'ymax': 470}},
 {'score': 0.9986783862113953,
  'label': 'cat',
  'box': {'xmin': 345, 'ymin': 23, 'xmax': 640, 'ymax': 368}}]
"""
