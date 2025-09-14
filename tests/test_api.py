import requests
import json

def test_api():
    try:
        response = requests.get('http://localhost:8000/health')
        print(f"Status Code: {response.status_code}")
        print("Response:")
        print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_api()
