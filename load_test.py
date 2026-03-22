import requests

url = "http://localhost:30007/predict"
image_path = "test_image.png"  # replace with your image file path

for i in range(200):
    with open(image_path, "rb") as f:
        files = {"file": f}
        response = requests.post(url, files=files)
        print(f"Request {i+1}: {response.status_code}, {response.json()}")