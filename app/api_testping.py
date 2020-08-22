import requests
import json

# api endpoint
endpoint_url = 'http://localhost:5000/predict'

#test file
img_path = '/home/archeron/dev/data/chest_xray/chest_xray/test/NORMAL/NORMAL2-IM-0278-0001.jpeg'

# payload
with open(img_path, "rb") as image:
	image = image.read()

payload = {"image": image}

# submit the request
r = requests.post(endpoint_url, files=payload).json()

# print response
print("file: " + img_path)
print(json.dumps(r, indent=4))
