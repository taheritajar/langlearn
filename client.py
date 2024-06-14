import os
import cv2
import numpy as np
import requests
import base64
import json
from PIL import Image
import io


# Make the API call
url = "http://127.0.0.1:8000/predict/"
file_path = r"C:\Users\Alireza\Desktop\arvr_proj\langlearn\temp_frames\frame_0006.jpg"
files = {'file': open(file_path, 'rb')}
response = requests.post(url, files=files)


# Check response
if response.status_code == 200:
    response_data = response.json()
    # print(response_data)
    print(json.dumps(response_data[0].get("predictions"), indent=4))

    image = cv2.imread(file_path)

    for obj_index, obj in enumerate(response_data[0].get("predictions")):
        print(obj_index)
        class_name = obj.get("class_name")
        bbox = obj.get("bbox")
        confidence = obj.get("confidence")

        # Draw bounding box
        color = list(np.random.random(size=3) * 250)
        cv2.rectangle(image, (bbox[0] + 10, bbox[1] + 10), (bbox[2] -10 , bbox[3] - 10), color, 2)
        label = f"{class_name}: {confidence:.2f}"
        cv2.putText(image, label, (bbox[0] + 40, bbox[1] + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # todo: image file name
    # Save the image to a file
    cv2.imwrite(f"{os.path.splitext(file_path)[0]}_result.jpg", image)

    # Show image result
    # window_name = 'image'
    # cv2.imshow(window_name, image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

else:
    print(f"Error: {response.status_code}")
    print(response.text)

