from datasets import load_dataset
dataset = load_dataset("BoJack/MMAR", split="test")
print(dataset.features)
print(dataset[0])