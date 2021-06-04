import requests
import sys
import json
from dialog_parser import get_requests

server = sys.argv[1]
if server.isdigit():
    server = "http://localhost:" + server
inputfile = sys.argv[2]
outputfile = sys.argv[3]
if(len(sys.argv) > 4):
    maxitems = int(sys.argv[4])
else:
    maxitems = 0

with open(inputfile) as infile:
    data = json.load(infile)

if isinstance(data,dict):
    data = get_requests(data)

if(maxitems > 0):    
    data = data[:maxitems]
        
result = requests.post(server,json=data).text
try:
    result = json.dumps(json.loads(result),indent=3)
except:
    pass

with open(outputfile,"w") as outfile:
    outfile.write(result)

