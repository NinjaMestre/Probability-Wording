from tqdm import tqdm
import numpy as np

SAMPLE_SPLIT = "\n"
START = "→"
END = "←"
SIZE = 64

with open("data.txt", "r", encoding="utf-8") as f:
    data_read = f.read()

chars = set(list(data_read))

samples = data_read.split(SAMPLE_SPLIT)
samples = [START + sample + END for sample in samples]

probabilities = {}

def make_sure_of_key(dict_, key, default):
    if key not in dict_.keys():
        dict_[key] = default

# Count occurrences
for sample_index, sample in tqdm(enumerate(samples), total=len(samples)):
    for character_index, character in enumerate(list(sample)):
        for other_index in range(character_index):
            other = sample[other_index]
            relative_index = other_index - character_index
            make_sure_of_key(probabilities, character, {})
            make_sure_of_key(probabilities[character], relative_index, {})
            make_sure_of_key(probabilities[character][relative_index], other, 0)
            probabilities[character][relative_index][other] += 1

# Normalize into probabilities
for character, indexes in probabilities.items():
    for index, others in indexes.items():
        _ = sum(probabilities[character][index].values())
        for other, chance in others.items():
            probabilities[character][index][other] /= _

# Inference
text = START
for i in tqdm(range(SIZE)):
    chances = {}
    for candidate in chars:
        make_sure_of_key(chances, candidate, 1)
        for character_index, character in enumerate(text):
            relative_index = character_index - len(text)
            try:
                chances[candidate] *= probabilities[candidate][relative_index][character]
            except Exception as e:
                chances[candidate] *= 0.000001
    
    _ = chances.values()
    _a = sum(_)
    if _a != 0:
        chances = {k: v / _a for k, v in chances.items()}
    
    try:
        new_char = np.random.choice(list(chances.keys()), p=list(chances.values()))
    except:
        new_char = END

    text += new_char

    if new_char == END:
        break

print(text)