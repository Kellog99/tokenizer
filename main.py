"""
Byte Pair Encoding (BPE) Implementation
A tokenization algorithm that iteratively merges the most frequent pair of bytes/characters
"""
from collections import Counter

import matplotlib.pyplot as plt

from tokenizer.wordpiece import WordPiece


def main():
    print("=" * 80)
    print("BYTE PAIR ENCODING DEMONSTRATION")
    print("=" * 80)
    print()

    # Generate test text
    print("Generating test text...")
    with open("./anelli.txt", "r") as f:
        train_text = f.read()
        train_text = train_text.lower()

    print(f"Test text length: {len(train_text)} characters")
    print(f"Test text words: {len(train_text.split())} words")
    print(f"Unique characters: {len(set(train_text.lower()))}")
    print()

    # Train BPE
    print("Training BPE with the whole text...")
    print("=" * 80)
    ita = {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
           "u", "v", "w", "x", "y", "z", "#", "!", '"', ":", ",", " ", ";", "-", "_", "0", "1", "2", "3", "4", "5", "6",
           "7", "8", "9", "\n", "\t", " "}
    bpe = WordPiece(
        alphabet=ita,
        max_iters=100,
        max_length=8
    )
    bpe.train(train_text)
    print(" train ended ".center(80, "#"))
    encoded_train = bpe.encode(train_text)
    tmp = Counter(encoded_train)
    print(f"total number of miss-tokenization = {tmp['[UNK]']}")
    print(f"ratio = {len(encoded_train) / len(train_text)}")

    print(bpe.get_tokens()[:100])
    with open("./i_promessi_sposi.txt", "r") as f:
        test_text = f.read()

    encoded_test = bpe.encode(test_text)
    print(f"ratio = {len(encoded_test) / len(test_text)}")

    cnt_train = Counter(encoded_train)
    cnt_test = Counter(encoded_test)

    plt.figure(figsize=(12, 5))

    # Plot both histograms with transparency for overlap visibility
    plt.hist(cnt_train.values(), density=True, alpha=0.6, bins=30, label='Train', color='steelblue', edgecolor='black')
    plt.hist(cnt_test.values(), density=True, alpha=0.6, bins=30, label='Test', color='coral', edgecolor='black')

    plt.title("Distribution of Character Frequencies: Train vs Test", fontsize=14, fontweight='bold')
    plt.xlabel("Character Frequency", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
