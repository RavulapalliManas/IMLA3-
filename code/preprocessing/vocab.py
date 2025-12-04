import os
import json
import pickle

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Look for itos.json in current directory or Pkl Files directory
itos_paths = [
    os.path.join(script_dir, "itos.json"),
    os.path.join(os.path.dirname(script_dir), "Pkl Files", "itos.json"),
]

itos_path = None
for path in itos_paths:
    if os.path.exists(path):
        itos_path = path
        break

if itos_path is None:
    print("Error: itos.json not found in any of the expected locations:")
    for path in itos_paths:
        print(f"  - {path}")
    exit(1)

# Load itos as a list
with open(itos_path, "r") as f:
    itos_list = json.load(f)

# Convert into proper itos dict
itos = {i: tok for i, tok in enumerate(itos_list)}

# Build stoi from itos
stoi = {tok: i for i, tok in itos.items()}

# Extract special token IDs
pad_id = stoi["<pad>"]
start_id = stoi["<start>"]
end_id = stoi["<end>"]

# Make vocab.pkl
vocab = {
    "stoi": stoi,
    "itos": itos,
    "pad_id": pad_id,
    "start_id": start_id,
    "end_id": end_id,
    "vocab_size": len(itos)
}

# Save to the same directory as itos.json was found
output_path = os.path.join(os.path.dirname(itos_path), "vocab.pkl")
with open(output_path, "wb") as f:
    pickle.dump(vocab, f)

print(f"Saved vocab.pkl to {output_path}")
