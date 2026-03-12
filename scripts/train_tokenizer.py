import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="C:/Users/Dell/OneDrive/Desktop/EduliteLLM/dataset.txt",
    model_prefix="C:/Users/Dell/OneDrive/Desktop/EduliteLLM/edulite_tokenizer",
    vocab_size=20000,
    character_coverage=1.0,
    model_type="bpe"
)

print("Tokenizer trained successfully!")