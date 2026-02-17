from datasets import load_dataset
ds = load_dataset("theoracle/Italian.sentiment.analysis", split='train')
print("Columns:", ds.column_names)
print("First 3 examples:")
for i in range(3):
    print(f"  {i}: {ds[i]}")