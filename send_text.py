import requests

print("Review: ")
input = input()

response = requests.post(
    url="http://127.0.0.1:5000/detection_server",
    json={
        "description": input,
        "userScore": 5
    },
)

print("Testing: ", response.json())
