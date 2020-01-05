import requests, os, sys
import json
model_id = os.environ.get('NANONETS_MODEL_ID')
api_key = os.environ.get('NANONETS_API_KEY')
image_path = sys.argv[1]

url = 'https://app.nanonets.com/api/v2/ObjectDetection/Model/' + model_id + '/LabelFile/'

data = {'file': open(image_path, 'rb'),    'modelId': ('', model_id)}

response = requests.post(url, auth=requests.auth.HTTPBasicAuth(api_key, ''), files=data)

# print(response.text)
json_data=response.text
parsed_json = (json.loads(json_data))
print(json.dumps(parsed_json, indent=4, sort_keys=True))
aligned=json.dumps(parsed_json, indent=4, sort_keys=True)
print(len(aligned["result"]["prediction"]))

# print(aligned["message"])

