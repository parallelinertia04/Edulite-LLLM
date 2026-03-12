import sentencepiece as spm

# load trained tokenizer
sp = spm.SentencePieceProcessor()
sp.load("C:/Users/Dell/OneDrive/Desktop/EduliteLLM/edulite_tokenizer.model")

with open("C:/Users/Dell/OneDrive/Desktop/EduliteLLM/dataset.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokens = sp.encode(text)
print("Total tokens:", len(tokens))

with open("C:/Users/Dell/OneDrive/Desktop/EduliteLLM/tokens.txt", "w") as f:
    f.write(" ".join(map(str, tokens)))
print("Dataset tokenized successfully!")