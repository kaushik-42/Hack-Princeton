import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5x6', pretrained=True)


"YOLOv5x6"
# Images
imgs = ['https://ultralytics.com/images/zidane.jpg']  # batch of images

img_1 = ['/Users/kaushiktummalapalli/Desktop/Image Processing/Hack-Princeton/vision/captioning/sai.jpeg']
# Inference
img_p = ['https://www.google.com/imgres?imgurl=https%3A%2F%2Fwww.olympiawa.gov%2Frevize_photo_gallery%2FServices%2FTransportation%2Ftraffic-signal-header.jpg&tbnid=zVhs_rxxVZFJRM&vet=12ahUKEwjcyL2o4Yf-AhXAF2IAHfISCoIQMygVegUIARDVAw..i&imgrefurl=https%3A%2F%2Fwww.olympiawa.gov%2Fservices%2Ftransportation%2Fsigns%2C_signals___streetlights.php&docid=1tm45jDr7A2z2M&w=1000&h=455&q=Traffic%20signs%20and%20lights%20images&ved=2ahUKEwjcyL2o4Yf-AhXAF2IAHfISCoIQMygVegUIARDVAw']
results = model(img_p)

# Results
results.print()
results.save()  # or .show()

results.xyxy[0]  # img1 predictions (tensor)

print(results.pandas().xyxy[0])  # img1 predictions (pandas)
#      xmin    ymin    xmax   ymax  confidence  class    name
# 0  749.50   43.50  1148.0  704.5    0.874023      0  person
# 1  433.50  433.50   517.5  714.5    0.687988     27     tie
# 2  114.75  195.75  1095.0  708.0    0.624512      0  person
# 3  986.00  304.00  1028.0  420.0    0.286865     27     tie
