import requests
import json

url = "https://sililar.up.railway.app/api/checker/files/8222aa33-a8f8-47f3-9d52-87b38dffcaf8/result"

payload = json.dumps({
  "file_check_id": "8222aa33-a8f8-47f3-9d52-87b38dffcaf8",
  "result": [
    {
      "file_resource_id": "0a58b8ab-60e7-42e3-b765-6d633a1f2d44",
      "result":  "sxxx"
    },
    {
      "file_resource_id": "159ba092-9ee0-451d-9663-d54722971154",
      "result":  "sxxx"
    }
  ],
  "create_at": "2024-06-12T15:04:05Z",
  "duration": 120
})
headers = {
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)
