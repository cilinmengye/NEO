import json
import os

cur_dir = os.path.dirname(os.path.realpath(__file__))
json_path = f"{cur_dir}/osc-Llama-2-7b-hf.json"

with open(json_path, "r") as f:
    configs = json.load(f)

maxlen = 0

for config in configs:
    maxlen = max(maxlen, int(config['prompt']) + int(config['max_tokens']))

print(maxlen)