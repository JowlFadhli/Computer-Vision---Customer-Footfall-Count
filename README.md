# Computer Vision: Customer Footfall Count

## Sidenote
To run the code, in prompt, using these requirement:
```python 'location_of_human_detection.py\human_detection.py' -v 'location_to_video' ```

## 1. Installing and Importing necessary library
Before running the code, necessary library should be installed in order to perform the tasks. The necessary library can be install using pip command:

```
pip install opencv-python
pip install imutils
pip install numpy
```

which later can be imported in the code respectively.

## 2. Model utilization for human detector
For intermediate accuracy result of the video, preliminary model are available to be applied which used HOGDescriptor with SVM implemented in OpenCV. To call the pre-trained model, below code is used to call Humand detection model in OpenCV to be feed into our SVM.

```
HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
```
## 3. Detecting on each frame of the video

In the code, as the detectByPathVideo function is called, detect_human function will detect human on every human on every frame given ```bounding_box_coordinates, weights = HOGCV.detectMultiScale(gray, winStride=(8, 8), padding=(8, 8), scale=1.03)``` can be adjusted and tuned to enhance the accuracy of the model.

## 4. Result
The outcome produced a model which can detect human with obvious movement but provides less to no detection for static object/human. Further parameter tuning can be applied to enhance the accuracy.

<img width="599" alt="image" src="https://user-images.githubusercontent.com/93107581/178884045-d254eb3b-8202-4cb4-a27c-f417fbf720bd.png">


References:
https://data-flair.training/blogs/python-project-real-time-human-detection-counting/ 
