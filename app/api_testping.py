import requests
import json

# api endpoint
endpoint_url = 'http://localhost:5001/predict'

#test file
img_path = '/home/ubuntu/pneumonia_detection_api/app/static/img/NORMAL2-IM-0173-0001-0001.jpeg'

# payload
with open(img_path, "rb") as image:
	image = image.read()

payload = {"image": image}

# submit the request
r = requests.post(endpoint_url, files=payload).json()

# print response
print("file: " + img_path)
print(json.dumps(r, indent=4))
