import abc
import json
import os.path
from collections import Counter
from typing import Optional

import numpy as np
from tqdm import tqdm

from utils.dfs import dfs
from utils.tree import TreeNode, add_branch


class Tokenizer:
    def __init__(
            self,
            alphabet: set[str],
            max_iters: int = 100,
            max_length: int = 4,
    ):
        """
        Initialize BPE with specified number of merge operations
        """
        self.alphabet = alphabet if alphabet else set()
        root: TreeNode = TreeNode(value="")  # Vocabulary of tokens
        for char in alphabet:
            root.add_child(char)
        self.dictionary = root

        self.max_iters = max_iters
        self.max_length = max_length

    def get_word(self, text: list[str], i: int) -> tuple[str, int]:
        node = self.dictionary
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
            node: TreeNode = self.dictionary
            word = ""

            while i < len(text) and node.is_child(text[i]):
                word += text[i]
                node = node.children[text[i]]
                i += 1

            # Guardrail in case new things are encountered
            if i < len(text) and text[i] not in self.dictionary.children:
                self.dictionary.add_child(text[i])

            out.append(word)

        return out

    def get_pair_freq(self, text: list[str]):
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

    @abc.abstractmethod
    def update(
            self,
            pair_freq: Optional[dict[tuple[str, str], float]] = None,
            freq: Optional[dict[str, float]] = None
    ) -> tuple[str, str]:
        """
        This represents the logic that every tokenizer has to implement for choosing the next union
        :return:
        """

    def train(self, text: str):
        """
        Train BPE on input text
        """

        # Check whether the original alphabet includes all the characters that are in the text
        if len(self.dictionary.children.keys()) == 0:
            text_char = set([c for c in text])
            text_char.update(self.alphabet)
            for w in text_char:
                self.dictionary.add_child(w)

        text = [c for c in text]
        # Build initial vocabulary
        best_dict = self.dictionary
        best_value = -np.inf

        pbar = tqdm(range(self.max_iters))
        for _ in pbar:
            encoded = self.encode(text)
            freq: dict[str, float] = Counter(encoded)
            pair_freq = self.get_pair_freq(text)
            length_dict = self.num_tokens()
            cross_entropy = sum(v / len(encoded) * np.log(v / len(encoded)) if v != 0 else 0 for v in freq.values())

            words = self.update(pair_freq=pair_freq, freq=freq)

            bayes = length_dict - cross_entropy * len(encoded)
            pbar.set_description(f"bayes: {bayes:.3f}, length: {length_dict}")
            # Early stopping
            if bayes > best_value:
                best_value = bayes
                best_dict = self.dictionary

            key = words[0] + words[1]

            add_branch(tree=self.dictionary, branch=[char for char in key])

        self.dictionary = best_dict

    def get_tokens(self) -> list[str]:
        out = list(self.dictionary.children.keys())
        bpe: list[str] = dfs(self.dictionary, "")
        out.extend(bpe)
        out = list(set(out))
        return out

    def num_tokens(self) -> int:
        return len(self.get_tokens())

    def save(self, path: str = "./pbe_weights.json"):
        json_file = {
            "alphabet": list(self.dictionary.children.keys()),
            "subword": dfs(self.dictionary, "")
        }
        if not path.endswith(".json"):
            path = path + "elements.json"

        with open(path, "w") as f:
            json.dump(json_file, f)

    def load(self, path: str):
        if os.path.exists(path):
            with open(path, "r") as f:
                json_file = json.load(f)
            root = TreeNode(value="")
            for el in json_file["alphabet"]:
                root.add_child(el)

            for word in json_file["subword"]:
                add_branch(root, word)

    def remove_unused_token(self, freq_encoded: dict[str, float]):
        token_used: list[str] = freq_encoded.keys()
        all_tokens = self.get_tokens()
        token_unused = set(all_tokens) - set(token_used)
        # The alphabet cannot be removed
        token_unused = token_unused - self.alphabet

        for token in token_unused:
            self.dictionary.remove_branch(branch=token)
