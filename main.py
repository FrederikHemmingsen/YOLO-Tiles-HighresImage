import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model (Many different models but m is larger than p but also slower)
model = YOLO("yolov8m.pt") 

# Class indices for person, truck, and boat in COCO dataset
# read about it here https://arxiv.org/pdf/1405.0312


TARGET_CLASSES = {0, 7, 9}

"""
1	person	person	person	person
2	bicycle	bicycle	bicycle	vehicle
3	car	car	car	vehicle
4	motorcycle	motorcycle	motorcycle	vehicle
5	airplane	airplane	airplane	vehicle
6	bus	bus	bus	vehicle
7	train	train	train	vehicle
8	truck	truck	truck	vehicle
9	boat	boat	boat	vehicle
10	traffic light	traffic light	traffic light	outdoor
11	fire hydrant	fire hydrant	fire hydrant	outdoor
12	street sign	-	-	outdoor
13	stop sign	stop sign	stop sign	outdoor
14	parking meter	parking meter	parking meter	outdoor
15	bench	bench	bench	outdoor
16	bird	bird	bird	animal
17	cat	cat	cat	animal
18	dog	dog	dog	animal
19	horse	horse	horse	animal
20	sheep	sheep	sheep	animal
21	cow	cow	cow	animal
22	elephant	elephant	elephant	animal
23	bear	bear	bear	animal
24	zebra	zebra	zebra	animal
25	giraffe	giraffe	giraffe	animal
26	hat	-	-	accessory
27	backpack	backpack	backpack	accessory
28	umbrella	umbrella	umbrella	accessory
29	shoe	-	-	accessory
30	eye glasses	-	-	accessory
31	handbag	handbag	handbag	accessory
32	tie	tie	tie	accessory
33	suitcase	suitcase	suitcase	accessory
34	frisbee	frisbee	frisbee	sports
35	skis	skis	skis	sports
36	snowboard	snowboard	snowboard	sports
37	sports ball	sports ball	sports ball	sports
38	kite	kite	kite	sports
39	baseball bat	baseball bat	baseball bat	sports
40	baseball glove	baseball glove	baseball glove	sports
41	skateboard	skateboard	skateboard	sports
42	surfboard	surfboard	surfboard	sports
43	tennis racket	tennis racket	tennis racket	sports
44	bottle	bottle	bottle	kitchen
45	plate	-	-	kitchen
46	wine glass	wine glass	wine glass	kitchen
47	cup	cup	cup	kitchen
48	fork	fork	fork	kitchen
49	knife	knife	knife	kitchen
50	spoon	spoon	spoon	kitchen
51	bowl	bowl	bowl	kitchen
52	banana	banana	banana	food
53	apple	apple	apple	food
54	sandwich	sandwich	sandwich	food
55	orange	orange	orange	food
56	broccoli	broccoli	broccoli	food
57	carrot	carrot	carrot	food
58	hot dog	hot dog	hot dog	food
59	pizza	pizza	pizza	food
60	donut	donut	donut	food
61	cake	cake	cake	food
62	chair	chair	chair	furniture
63	couch	couch	couch	furniture
64	potted plant	potted plant	potted plant	furniture
65	bed	bed	bed	furniture
66	mirror	-	-	furniture
67	dining table	dining table	dining table	furniture
68	window	-	-	furniture
69	desk	-	-	furniture
70	toilet	toilet	toilet	furniture
71	door	-	-	furniture
72	tv	tv	tv	electronic
73	laptop	laptop	laptop	electronic
74	mouse	mouse	mouse	electronic
75	remote	remote	remote	electronic
76	keyboard	keyboard	keyboard	electronic
77	cell phone	cell phone	cell phone	electronic
78	microwave	microwave	microwave	appliance
79	oven	oven	oven	appliance
80	toaster	toaster	toaster	appliance
81	sink	sink	sink	appliance
82	refrigerator	refrigerator	refrigerator	appliance
83	blender	-	-	appliance
84	book	book	book	indoor
85	clock	clock	clock	indoor
86	vase	vase	vase	indoor
87	scissors	scissors	scissors	indoor
88	teddy bear	teddy bear	teddy bear	indoor
89	hair drier	hair drier	hair drier	indoor
90	toothbrush	toothbrush	toothbrush	indoor
91	hair brush	-	-	indoor
"""


def detect_objects(image, conf_threshold=0.3, iou_threshold=0.4):
    results = model(image)
    detections = results[0].boxes  # Extract boxes from the first image in the batch
    return [
        (int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3]), float(box.conf[0]), int(box.cls[0]))
        for box in detections if float(box.conf[0]) >= conf_threshold and int(box.cls[0]) in TARGET_CLASSES
    ]

def tile_image(image, tile_size=1024, overlap=128):
    height, width, _ = image.shape
    tiles = []
    for y in range(0, height, tile_size - overlap):
        for x in range(0, width, tile_size - overlap):
            tile = image[y:y + tile_size, x:x + tile_size]
            tiles.append((tile, x, y))
    return tiles

def process_image(image, tile_size=1024, overlap=128, conf_threshold=0.3, iou_threshold=0.4):
    tiles = tile_image(image, tile_size, overlap)
    all_detections = []

    for tile, x_offset, y_offset in tiles:
        detections = detect_objects(tile, conf_threshold, iou_threshold)
        for (x1, y1, x2, y2, confidence, class_id) in detections:
            all_detections.append((x1 + x_offset, y1 + y_offset, x2 + x_offset, y2 + y_offset, confidence, class_id))

    return all_detections

def draw_detections(image, detections, classes):
    for (x1, y1, x2, y2, confidence, class_id) in detections:
        label = f"{classes[class_id]} {confidence:.2f}"
        color = (0, 255, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Load image
image_path = "high_res_image.jpg"
image = cv2.imread(image_path)

# Check if the image was successfully loaded
if image is None:
    print(f"Error: Unable to open image file {image_path}")
else:
    # Process the image with adjusted thresholds
    detections = process_image(image, conf_threshold=0.3, iou_threshold=0.4)

    # Load class labels
    classes = model.names

    # Draw the detections on the original image
    draw_detections(image, detections, classes)

    # Convert BGR image (OpenCV format) to RGB (matplotlib format)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Save the output image
    output_path = "output_image.jpg"
    cv2.imwrite(output_path, image)
    print(f"Output saved to {output_path}")


    # Display the image using matplotlib
    import matplotlib.pyplot as plt
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()

  