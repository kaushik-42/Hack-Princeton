import cv2
from PIL import Image
from transformers import pipeline


class ObjectDetector:
    def __init__(self, model_name='object-detection', image_classes=['traffic light', 'traffic sign', 'person']):
        self.image_classes = image_classes
        self.object_detector = pipeline(model_name)

    def detect_objects(self, image_path):
        image = Image.open(image_path)
        results = self.object_detector(image)
        filtered_results = [
            obj for obj in results if obj['label'] in self.image_classes]
        return filtered_results

    def save_bounding_boxes(self, image_path):
        results = self.detect_objects(image_path)
        img = cv2.imread(image_path)
        for i, result in enumerate(results):
            xmin = result['box']['xmin']
            ymin = result['box']['ymin']
            xmax = result['box']['xmax']
            ymax = result['box']['ymax']
            label = result['label']

            cropped_img = img[ymin:ymax, xmin:xmax]
            file_name = label.replace(" ", "_") + "_" + str(i+1) + ".jpeg"
            cv2.imwrite('object_detection/data_results/'+file_name, cropped_img)


# object_detector = ObjectDetector()
# object_detector.save_bounding_boxes(
#     'download.jpeg')

"""
import cv2
import requests
from PIL import Image
from transformers import pipeline
# from io import BytesIO

# Download an image with cute cats
url = "/Users/kaushiktummalapalli/Desktop/Image Processing/Hack-Princeton/vision/captioning/sai.jpeg"
# image_data = requests.get(url, stream=True).raw
image = Image.open(url)

# Allocate a pipeline for object detection
object_detector = pipeline('object-detection')

# print(object_detector(image))
result = object_detector(image)
image_classes = ['traffic light', 'traffic sign', 'person']
filtered_results = [obj for obj in result if obj['label'] in image_classes]

# Loop through each filtered result and create an image for each bounding box
for i, result in enumerate(filtered_results):
    xmin = result['box']['xmin']
    ymin = result['box']['ymin']
    xmax = result['box']['xmax']
    ymax = result['box']['ymax']
    label = result['label']

    # Crop the original image to get the bounding box image
    # bbox_image = image.crop((xmin, ymin, xmax, ymax))

    # Save the bounding box image to a file
    # bbox_image.save(f"bounding_box_{i}.jpg")
    img = cv2.imread(url)
    cropped_img = img[ymin:ymax, xmin:xmax]
    file_name = label.replace(" ", "_") + "_" + str(i+1) + ".jpg"
    cv2.imwrite(file_name, cropped_img)




# OLD 


# Load the image
img = cv2.imread(
    "/Users/kaushiktummalapalli/Desktop/Image Processing/Hack-Princeton/vision/captioning/p.jpeg")

# Loop through each object and draw the corresponding bounding box
for obj in filtered_helper:
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


# Results and bounding boxes for the labels we generated:
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
