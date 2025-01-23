import pytest
import requests


def test_api():
    url = "https://sentiment-predict-727127387938.europe-west1.run.app/predict"
    review = "This is a great app, I love it!"
    response = requests.post(url, json={"review": review})

    assert response.status_code == 200, f"Expected status code 400, but got {response.status_code}"

    assert response.json()['score'] < 1, f"Score cannot be greater than 1"
    assert response.json()['score'] >0, f"Score cannot be smaller than 0"
