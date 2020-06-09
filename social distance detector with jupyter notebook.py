#!/usr/bin/env python
# coding: utf-8

# # importing required packages

# In[1]:


import numpy as np
import cv2
import matplotlib.pyplot as plt
from math import sqrt
from utils.functions import *
from utils.view import *


# # Calibration

# importing the image file from multimedia directory
# In this cell we will be selecting 4 points manually in the image and will be using the sidewalk as a reference. We will draw the selected points on the image, the connection lines between the points to form the trapezoid and then generate the image

# In[2]:


original = cv2.cvtColor(cv2.imread('multimedia/calibration_frame.png'), cv2.COLOR_BGR2RGB)
image_calibration = original.copy()
image_copy = original.copy()

source_points = np.float32([[1187, 178], [1575, 220], [933,883], [295, 736]])


for point in source_points:
    cv2.circle(image_calibration, tuple(point), 8, (255, 0, 0), -1)


points = source_points.reshape((-1,1,2)).astype(np.int32)
cv2.polylines(image_calibration, [points], True, (0,255,0), thickness=4)


print("Image size: ", original.shape)
plt.figure(figsize=(12, 12))
plt.imshow(image_calibration)
plt.show()


# Considering the final size of the image as 4500x3000 pixels. 
# After obtaining the reference points of the original image, we calculate the homography matrix for the transformation

# # Homography Matrix Calculation

# In[3]:


dst_size=(4500,3000)

src=np.float32([[1187, 178], [1575, 220], [933,883], [295, 736]])

dst=np.float32([(0.57,0.42), (0.65, 0.42), (0.65,0.84), (0.57,0.84)])

img_size = np.float32([(image_calibration.shape[1],image_calibration.shape[0])])

dst = dst * np.float32(dst_size)

H_matrix = cv2.getPerspectiveTransform(src, dst)
print("The perspective transform matrix:")
print(H_matrix)


# Applying warp perspective to transform the image with the obtained homography matrix and generate the transformed image

# # Bird Eye View

# In[4]:


warped = cv2.warpPerspective(image_calibration, H_matrix, dst_size)

plt.figure(figsize=(12, 12))
plt.imshow(warped)
plt.xticks(np.arange(0, warped.shape[1], step=150))
plt.yticks(np.arange(0, warped.shape[0], step=150))
plt.grid(True, color='g', linestyle='-', linewidth=0.9)
plt.show()


# # Detection

# The YOLOv3-608 architecture with a map of 57.9 and 20 FPS processing was chosen because it is desired to have the closest processing to be performed in real time. The minimum distance value was chosen as 115 pixels after several trial and error tests.

# firsty we define the confidentiality parameters for CNN.
# then setting the minimum distance between the persons to be 115 pixels ~ 1.1979166666666667 inches.
# after importing the configuration files of CNN and creating calsses for those configuration files we will create a CNN Model to get the detection layers

# In[5]:


confidence_threshold = 0.5
nms_threshold = 0.4


min_distance = 115
width = 608
height = 608

config = 'models/yolov3.cfg'
weights = 'models/yolov3.weights'
classes = 'models/coco.names'

with open(classes, 'rt') as f:
    coco_classes = f.read().strip('\n').split('\n')

model = create_model(config, weights)
output_layers = get_output_layers(model)


# # Human Detection and Assessment of Minimum Distance between them.

# blob_from_image(): Detection boxes are obtained
# non_maximum_suppression(): is a prediction function which is applied to remove false positives. 
# people_distances_bird_eye_view(): Each lower center point corresponding to the detection boxes is transformed into the "Bird's Eye View".
# The distance between the transformed points in the "Bird's Eye View" is evaluated, taking this as a reference a distance more than 115 pixels to be considered as dangerous.
# The original image is graphed by coloring the detection boxes as red if the calculated distance is more than 115 pixels.

# In[6]:


blob = blob_from_image(image_copy, (width, height))

outputs = predict(blob, model, output_layers)

boxes, nms_boxes, class_ids = non_maximum_suppression(image_copy, outputs, confidence_threshold, nms_threshold)

person_boxes = get_domain_boxes(coco_classes, class_ids, nms_boxes, boxes, domain_class='person')

good, bad = people_distances_bird_eye_view(person_boxes, min_distance)

new_image  = draw_new_image_with_boxes(image_copy, good, bad, min_distance, draw_lines=True)
plt.figure(figsize=(12,12))
plt.imshow(new_image)
plt.show()


# The "Bird's Eye View" is graphed by coloring the dots as red if the calculated distance is more than 115 pixels and green otherwise.

# In[7]:



green_points = [g[6:] for g in good]
red_points = [r[6:] for r in bad]


bird_eye_view = generate_bird_eye_view(green_points, red_points)
plt.figure(figsize=(6,6))
plt.imshow(bird_eye_view)
plt.show()


# It shows the final image that includes the original view with the detection boxes and the colored points according to the distance evaluation in the original image and the "Bird's Eye View" view respectively.

# In[8]:



picture = generate_picture()

img_final = generate_content_view(picture, new_image, bird_eye_view)
plt.figure(figsize=(20,20))
plt.imshow(img_final)
plt.show()


# # Video Processing

# Finally, in this last section, the code will perform the processing of the video "test.mp4" is presented, applying the same steps previously shown for each of the video frames.

# In[10]:


get_ipython().run_cell_magic('time', '', 'confidence_threshold = 0.5\nnms_threshold = 0.4\nmin_distance = 115\nwidth = 608\nheight = 608\n\nconfig = \'models/yolov3.cfg\'\nweights = \'models/yolov3.weights\'\nclasses = \'models/coco.names\'\n\nwith open(classes, \'rt\') as f:\n    coco_classes = f.read().strip(\'\\n\').split(\'\\n\')\n\nwriter = None\nW = 2604\nH = 1254    \n  \nmodel = create_model(config, weights)\noutput_layers = get_output_layers(model)\n\npicture = generate_picture()\n\nvideo = cv2.VideoCapture(\'multimedia/test.mp4\')\n\nwhile True:\n  \n  _,frame = video.read()\n  \n  if frame is None:\n      break\n  \n  if writer is None:\n    fourcc = cv2.VideoWriter_fourcc(*"MJPG")\n    writer = cv2.VideoWriter(\'results/output.avi\', fourcc, 30, (W, H), True)\n\n  image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)\n  blob = blob_from_image(image, (width, height))\n  outputs = predict(blob, model, output_layers)\n  boxes, nms_boxes, class_ids = non_maximum_suppression(image, outputs, confidence_threshold, nms_threshold)\n  person_boxes = get_domain_boxes(coco_classes, class_ids, nms_boxes, boxes, domain_class=\'person\')\n\n  good, bad = people_distances_bird_eye_view(person_boxes, min_distance)\n  new_image  = draw_new_image_with_boxes(image, good, bad, min_distance, draw_lines=True)\n  \n  green_points = [g[6:] for g in good]\n  red_points = [r[6:] for r in bad]\n\n  bird_eye_view = generate_bird_eye_view(green_points, red_points)\n  output_image = generate_content_view(picture, new_image, bird_eye_view)\n\n  if writer is not None:\n    plt.figure(figsize=(20,20))\n    plt.imshow(output_image)\n    plt.show()  \n    writer.write(output_image[:,:,::-1])\n      \nif writer is not None:\n  writer.release()\n\nvideo.release()')


# In[ ]:




