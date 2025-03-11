import json

with open("./hh", "r") as f:
    for line in f:
        try:
            loaded = json.loads(line)
            break
        except:
            pass

print(loaded)