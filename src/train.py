import torch

with open("data/wizard_of_oz.txt", "r", encoding="utf-8") as f:
    text = f.read()

print(text[:1000])
print()
print("Total characters:", len(text))