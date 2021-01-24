import requests
import json 
import pandas as pd

#df = pd.read_csv("myfile.csv")
with open('annotated.png', 'rb') as f:
    r = requests.post('http://httpbin.org/post', files={'annotated.png': f})
#data = {}
#data["user_name"] = df.to_dict()
#headers = {'content-type': 'application/json'}
#url = 'http://localhost:8000/'
#resp = requests.post(url,data=json.dumps(data), headers=headers )
#resp