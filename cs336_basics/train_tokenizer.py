import cProfile
import pstats
from cs336_basics.tokenizer import train_bpe
import json

def main() -> None:
    vocab, merges = train_bpe(
        input_path="data/TinyStoriesV2-GPT4-valid.txt",
        vocab_size=1000,
        special_tokens=["<|endoftext|>"],
        num_processes=8,
    )

    for tok1, tok2 in merges:
        with open("data/TinyStoriesV2-GPT4-valid-merges.txt",'a') as merges_file:
                merges_file.write(f"{tok1.hex()} {tok2.hex()} \n")

    hex_vocab = {}
    for index, token_bytes in vocab.items():
        hex_vocab[token_bytes.hex()] = index

    with open("data/TinyStoriesV2-GPT4-valid-vocab.json",'w') as vocab_file:
        json.dump(hex_vocab,vocab_file)


    print(f"Vocabulary size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")


if __name__ == "__main__":
    profiler = cProfile.Profile()

    profiler.enable()
    main()
    profiler.disable()

    profiler.dump_stats("train_bpe.prof")

    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats("cumulative")
    stats.print_stats(40)



