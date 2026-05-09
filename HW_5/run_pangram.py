"""
Code for HW5 CMSC 25800 Spring 2026
"""
from pangram import Pangram
import json

INPUT_FILE = "essay_1.txt"
OUTPUT_FILE = "hw5_results.jsonl"

pangram_client = Pangram(api_key="bab59582-f372-4db2-a06c-9875d4ea0a02")

with open(INPUT_FILE, "r") as f:
    text = f.read()

result = pangram_client.predict(text)

with open(OUTPUT_FILE, "a") as f:
    f.write(f"{json.dumps(result)}\n")