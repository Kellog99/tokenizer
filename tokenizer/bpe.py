from typing import Optional

import numpy as np
from tqdm import tqdm

from tokenizer.tree import TreeNode, dfs


class BytePairEncoding:
    def __init__(
            self,
            alphabet: Optional[set[str]] = None,
            max_iters: int = 100,
            max_length: int = 8,
    ):
        """
        Initialize BPE with specified number of merge operations
        """
        self.merges = []  # List of (pair, new_token) tuples
        self.alphabet: TreeNode = TreeNode(value="")  # Vocabulary of tokens
        if alphabet:
            self.alphabet.next = {a: TreeNode(value=a) for a in alphabet}

        self.max_iters = max_iters
        self.max_length = max_length

    def get_word(self, text: list[str], i: int) -> tuple[str, int]:
        node = self.alphabet
        out = ""
        while i < len(text) and node.is_child(text[i]):
            node = node.children[text[i]]
            out += text[i]
            i += 1
            if i < len(text):
                print(i, text[i], len(text))

        return out, i

    def encode(self, text: list[str] | str) -> list[str]:
        if isinstance(text, str):
            text = [c for c in text]
        out = []
        i = 0
        while i < len(text):
            node = self.alphabet
            word = ""
            while i < len(text) and node.is_child(text[i]):
                word += text[i]
                node = node.children[text[i]]
                i += 1

            # Guardrail in case new things are encountered
            if i < len(text) and text[i] not in self.alphabet.children:
                self.alphabet.add_child(text[i])

            out.append(word)

        return out

    def get_freq(self, text: list[str]):
        freq: dict[tuple[str, str], int] = {}
        text_encoded = self.encode(text)

        for i in range(1, len(text_encoded)):
            a = text_encoded[i - 1]
            b = text_encoded[i]
            if len(a + b) <= self.max_length:
                if (a, b) in freq:
                    freq[(a, b)] += 1
                else:
                    freq[(a, b)] = 1
        return {key: value for key, value in freq.items()}

    def train(self, text: str):
        """
        Train BPE on input text
        """

        # Convert words to character sequences with end-of-word marker
        if len(self.alphabet.children.keys()) == 0:
            ita = {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
                   "u", "v", "w", "x", "y", "z"}
            text_char = set([c for c in text])
            text_char.update(ita)
            for w in text_char:
                self.alphabet.add_child(w)

        text = [c for c in text]
        # Build initial vocabulary
        best_dict = self.alphabet
        best_value = -np.inf

        for _ in tqdm(range(self.max_iters)):
            freq = self.get_freq(text)

            encoded = self.encode(text)
            length_dict = self.num_tokens()

            probs = [v / len(encoded) * np.log(v / len(encoded)) if v != 0 else 0 for v in freq.values()]
            bayes = length_dict - sum(probs) * len(encoded)
            if bayes > best_value:
                best_value = bayes
                best_dict = self.alphabet
            tqdm.write(f"object = {bayes}, len dict = {length_dict}")

            words, max_freq = max(freq.items(), key=lambda x: x[1])

            tqdm.write(f"{words} = {max_freq}")
            if max_freq > 1:
                key = words[0] + words[1]

                def add_leaf(tree: TreeNode, path: list[str]):
                    if len(path) > 0:
                        if not tree.is_child(path[0]):
                            tree.add_child(path[0])

                        add_leaf(tree.get_child(path[0]), path=path[1:])

                add_leaf(tree=self.alphabet, path=[char for char in key])
            else:
                print("ending the loop")
                break

        self.alphabet = best_dict

    def get_tokens(self) -> list[str]:
        out = list(self.alphabet.children.keys())
        bpe: list[str] = dfs(self.alphabet, "")
        out.extend(bpe)
        out = list(set(out))
        return out

    def num_tokens(self) -> int:
        return len(self.get_tokens())
