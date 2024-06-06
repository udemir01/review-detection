import requests

response = requests.post(
    url="http://127.0.0.1:5000/detection_server",
    json={
        "description": "I really like this place",
        "userScore": 5
    },
)

print(response.json())
