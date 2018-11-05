import json
from pprint import pprint

with open('data/tag2label.json','rb') as f:
    a = json.load(f)

pprint(a["O"])