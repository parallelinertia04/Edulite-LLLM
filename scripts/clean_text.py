import os
import re

input_folder = "C:/Users/Dell/OneDrive/Desktop/EduliteLLM/dataset_text"
output_file = "C:/Users/Dell/OneDrive/Desktop/EduliteLLM/dataset.txt"

all_text = []

for file in os.listdir(input_folder):

    if file.endswith(".txt"):

        with open(os.path.join(input_folder, file), "r", encoding="utf-8") as f:

            text = f.read()

            text = text.lower()

            # remove extra spaces
            text = re.sub(r'\s+', ' ', text)

            # split sentences
            sentences = re.split(r'[.!?]', text)

            for s in sentences:
                s = s.strip()
                if len(s) > 20:
                    all_text.append(s)

with open(output_file, "w", encoding="utf-8") as f:
    for sentence in all_text:
        f.write(sentence + "\n")

print("Dataset cleaned and sentence-split successfully!")