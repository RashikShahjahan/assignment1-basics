import cProfile
import pstats
from cs336_basics.tokenizer import train_bpe, Tokenizer
import json
import numpy as np

def main() -> None:
    """
    vocab, merges = train_bpe(
        input_path="data/TinyStoriesV2-GPT4-train.txt",
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
        num_processes=16,
    )
    with open("data/TinyStoriesV2-GPT4-train-merges.txt", "w") as f:
        for tok1, tok2 in merges:
            f.write(f"{tok1.hex()} {tok2.hex()}\n")


    hex_vocab = {}
    for index, token_bytes in vocab.items():
        hex_vocab[token_bytes.hex()] = index

    with open("data/TinyStoriesV2-GPT4-train-vocab.json",'w') as vocab_file:
        json.dump(hex_vocab,vocab_file)
"""

    with open("data/TinyStoriesV2-GPT4-train-merges.txt") as f:
        merges = f.read()
    with open("data/TinyStoriesV2-GPT4-train-vocab.json") as f:
        hex_vocab = json.load(f)


    vocab = {
        token_id: bytes.fromhex(token_hex)
        for token_hex, token_id in hex_vocab.items()
    }
    tokenizer=Tokenizer(vocab,merges,["<|endoftext|>"])



    with open("data/TinyStoriesV2-GPT4-train.txt", encoding='utf-8') as f:
        train_tokens = np.fromiter(tokenizer.encode_iterable(f),dtype=np.uint16)
        np.save("data/train.npy",train_tokens)



    
    with open("data/TinyStoriesV2-GPT4-valid.txt", encoding='utf-8') as f:
        valid_tokens = np.fromiter(tokenizer.encode_iterable(f),dtype=np.uint16)
        np.save("data/valid.npy",valid_tokens)


    print(f"Vocabulary size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")


if __name__ == "__main__":
    """
    profiler = cProfile.Profile()

    profiler.enable()
    """
    main()
    """
    profiler.disable()

    profiler.dump_stats("train_bpe.prof")

    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats("cumulative")
    stats.print_stats(40)

    """

