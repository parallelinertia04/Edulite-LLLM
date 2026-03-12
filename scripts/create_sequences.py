import numpy as np

seq_length = 128

with open("C:/Users/Dell/OneDrive/Desktop/EduliteLLM/tokens.txt") as f:
    tokens = list(map(int, f.read().split()))

print("Total tokens:", len(tokens))

inputs = []
targets = []

for i in range(len(tokens) - seq_length):

    input_seq = tokens[i:i+seq_length]
    target_seq = tokens[i+1:i+seq_length+1]

    inputs.append(input_seq)
    targets.append(target_seq)

inputs = np.array(inputs)
targets = np.array(targets)

np.save("C:/Users/Dell/OneDrive/Desktop/EduliteLLM/inputs.npy", inputs)
np.save("C:/Users/Dell/OneDrive/Desktop/EduliteLLM/targets.npy", targets)

print("Training sequences created successfully!")