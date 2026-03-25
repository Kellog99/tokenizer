"""
Byte Pair Encoding (BPE) Implementation
A tokenization algorithm that iteratively merges the most frequent pair of bytes/characters
"""
from tokenizer.bpe import BytePairEncoding


def main():
    print("=" * 80)
    print("BYTE PAIR ENCODING DEMONSTRATION")
    print("=" * 80)
    print()

    # Generate test text
    print("Generating test text...")
    with open("./commedia.txt", "r") as f:
        train_text = f.read()

    print(f"Test text length: {len(train_text)} characters")
    print(f"Test text words: {len(train_text.split())} words")
    print(f"Unique characters: {len(set(train_text.lower()))}")
    print()

    # Train BPE
    print("Training BPE with the whole text...")
    print("=" * 80)
    bpe = BytePairEncoding(
        alphabet=set(),
        max_iters=150
    )
    bpe.train(train_text.lower())
    encoded = bpe.encode(train_text.lower())

    print(f"ratio = {len(encoded) / len(train_text)}")
    print(" train ended ".center(80, "#"))

    print(bpe.get_tokens())
    with open("./i_promessi_sposi.txt", "r") as f:
        test_text = f.read()

    encoded = bpe.encode(test_text)
    print(len(encoded))
    print(f"ratio = {len(encoded) / len(test_text)}")


if __name__ == "__main__":
    main()
