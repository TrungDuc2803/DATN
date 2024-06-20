import cv2
import numpy as np
from ultrafastLaneDetector import UltrafastLaneDetector, ModelType
import requests

model_path = "models/tusimple_18.pth"
model_type = ModelType.TUSIMPLE
use_gpu = False

# Initialize lane detection model
lane_detector = UltrafastLaneDetector(model_path, model_type, use_gpu)

# Initialize webcam
# cap = cv2.VideoCapture("D:/Hoc/DATN/Youtube/lanes detection files/DataInput/lane3_Input.mp4")
cap = cv2.VideoCapture(0)
cv2.namedWindow("Detected lanes", cv2.WINDOW_NORMAL)

frame_count = 0
skip_frames = 3
checkforward = True
checkleft = True
checkright = True
checkbackward = True
checkstop = True
def call_api(endpoint):
    global front_Distance, behind_Distance, left_Distance, right_Distance
    url = f'http://192.168.1.112:5000/{endpoint}'
    response = requests.get(url)
    if response.status_code == 200:
    	if endpoint == 'Distance':
            data = response.json()
            front_Distance = data['front_Distance']
            behind_Distance = data['behind_Distance']
            right_Distance = data['right_Distance']
            left_Distance = data['left_Distance']

        # print(f'Success calling {endpoint}:', response.status_code)
    else:
        print(f'Failed calling {endpoint}:', response.status_code)

def runforward():
    global checkforward
    if checkforward:
        call_api('Forward')
        checkforward = False

def runbackward():
    global checkbackward
    if checkbackward:
        call_api('Backward')
        checkbackward = False

def runleft():
    global checkleft
    if checkleft:
        call_api('TurnLeft')
        checkleft = False

def runright():
    global checkright
    if checkright:
        call_api('TurnRight')
        checkright = False

def stop():
    global checkforward, checkright, checkleft, checkbackward, checkstop
    if checkstop :
        call_api('Stop')
        checkforward = True
        checkright = True
        checkleft = True
        checkbackward = True
        checkstop = False

def Distance():
    call_api('Distance')


while True:
    ret, frame = cap.read()
    if not ret:
        break
    if frame_count % (skip_frames + 1) == 0:
        output_img, lanes_points = lane_detector.detect_lanes(frame)

        height, width = output_img.shape[:2]
        for lane_num, lane_points in enumerate(lanes_points):
            if lane_num == 1:
                max_y_point = min(lane_points, key=lambda point: point[1])

                # Draw arrowed line from center bottom to the lane point with max Y
                cv2.arrowedLine(output_img, (width // 2, height), tuple(max_y_point), (0, 255, 0), 2, tipLength=0.01)

                # Calculate vectors
                vector1 = np.array([max_y_point[0] - width // 2, max_y_point[1] - height])  # Vector from center bottom to lane point
                vector2 = np.array([width - width // 2, height - height])  # Vector from center bottom to (width, height)
# Calculate the angle between the vectors
                unit_vector1 = vector1 / np.linalg.norm(vector1)
                unit_vector2 = vector2 / np.linalg.norm(vector2)
                dot_product = np.dot(unit_vector1, unit_vector2)
                angle = np.arccos(dot_product)
                angle_degrees = np.degrees(angle)

                # Display the angle on the image
                cv2.putText(output_img, f'Angle: {angle_degrees:.2f} degrees', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if int(angle_degrees) > 70 and int(angle_degrees) < 110:
                    cv2.putText(output_img, f'forward', (width // 2, height - 30), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 255, 0), 2)
                    runforward()
                    checkleft = True
                    checkright = True

                elif int(angle_degrees) <= 70:
                    cv2.putText(output_img, f'right', (width // 2, height - 30), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 255, 0), 2)
                    runright()
                    checkleft = True
                    checkforward = True
                else:
                    cv2.putText(output_img, f'left', (width // 2, height - 30), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 255, 0), 2)
                    runleft()
                    checkforward = True
                    checkright = True
        # Draw line from center bottom to (width, height)
        cv2.line(output_img, (width // 2, height), (width // 2, 0), (0, 255, 0), 2)

        cv2.imshow("Detected lanes", output_img)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
stop()
cv2.destroyAllWindows()