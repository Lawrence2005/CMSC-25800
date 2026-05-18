"""
Code for HW5 CMSC 25800 Spring 2026
"""
from pangram import Pangram
import json

# METHOD_NUM, VARIATION_NUM, ESSAY_NUM = 3, 3, 3

# INPUT_FILE = f"M{METHOD_NUM}_V{VARIATION_NUM}_E{ESSAY_NUM}.txt"
# OUTPUT_FILE = f"M{METHOD_NUM}_V{VARIATION_NUM}.jsonl"

INPUT_FILE = "miscellaneous/essay_test6.txt"
OUTPUT_FILE = "miscellaneous/test.jsonl"

pangram_client = Pangram(api_key="bab59582-f372-4db2-a06c-9875d4ea0a02")

with open(INPUT_FILE, "r") as f:
    text = f.read()

result = pangram_client.predict(text)

with open(OUTPUT_FILE, "a") as f:
    f.write(f"{json.dumps(result, indent=4)}\n")