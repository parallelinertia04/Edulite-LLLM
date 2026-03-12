import sentencepiece as spm

# load tokenizer
sp = spm.SentencePieceProcessor()
sp.load("C:/Users/Dell/OneDrive/Desktop/EduliteLLM/edulite_tokenizer.model")

# test sentence
text = "A cell is the smallest unit of a living thing"

# encode text → tokens
tokens = sp.encode(text)

print("Input text:")
print(text)

print("\nToken IDs:")
print(tokens)

print("\nDecoded back to text:")
print(sp.decode(tokens))