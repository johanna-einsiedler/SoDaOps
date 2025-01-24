import requests

url = "https://ci-cd-api-727127387938.europe-west1.run.app"
payload = {"review": "I'm testing the alert"}

for _ in range(1000):
    r = requests.get(url, params=payload)
