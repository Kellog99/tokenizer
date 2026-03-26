import abc
import json
import os.path
import re
from collections import Counter
from typing import Optional

import numpy as np
from tqdm import tqdm

from utils.dfs import dfs
from utils.tree import TreeNode, add_branch, remove_branch, is_branch


class Tokenizer:
    def __init__(
            self,
            alphabet: set[str],
            max_iters: int = 100,
            max_length: int = 4,
            special_characters: set = {"\n", "\t", " "},
    ):
        """
        Initialize BPE with specified number of merge operations
        """
        self.alphabet = alphabet if alphabet else set()
        self.alphabet.update(special_characters)
        self.special_characters = special_characters

        root: TreeNode = TreeNode(key="")  # Vocabulary of tokens
        for char in alphabet:
            root.add_child(
                TreeNode(
                    key=char,
                    is_word_leaf=True,
                )
            )
        self.dictionary = root

        self.max_iters = max_iters
        self.max_length = max_length

    def basic_tokenizer(self, text: str) -> list[str]:
        # Capture words and whitespace separately
        tokens = re.findall(r'\S+|\n|\t| ', text)
        return tokens

    @abc.abstractmethod
    def decode(self, encoded_text: list[str]) -> str:
        pass

    @abc.abstractmethod
    def encode(self, text: str) -> list[str]:
        pass

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
        pass

    def concat_words(self, word_a: str, word_b: str) -> str:
        """
        This function represent the logic for which two word are concat.
        """
        return word_a + word_b

    def are_concatenable(self, word_a: str, word_b: str) -> bool:
        return True

    def get_pair_freq(self, text: str) -> dict[tuple[str, str], int]:
        text_encoded = self.encode(text)
        freq: dict[tuple[str, str], int] = {}

        for i in range(1, len(text_encoded)):
            a = text_encoded[i - 1]
            b = text_encoded[i]
            if self.are_concatenable(a, b):
                concat_word = self.concat_words(a, b)
                if len(concat_word) <= self.max_length:
                    if (a, b) in freq:
                        freq[(a, b)] += 1
                    else:
                        freq[(a, b)] = 1
        return freq

    def check_vocabulary(self, text: str):
        """
        This method check whether the alphabet has to be updated or, if needed, instanciate a better dictionary
        """
        text_char = set(text)
        # adding the remaining characters that are not written
        for w in list(text_char - self.alphabet):
            self.dictionary.add_child(
                TreeNode(
                    key=w,
                    is_word_leaf=True,
                )
            )
        # Updating the alphabet
        self.alphabet.update(text_char)

    def train(self, text: str):
        """
        This function represent the training procedure for a tokenizer
        """

        # Check whether the original alphabet includes all the characters that are in the text
        self.check_vocabulary(text)

        # Build initial vocabulary
        best_dict = self.dictionary
        best_value = -np.inf
        ######## starting point ##########
        # Encoding
        encoded = self.encode(text)
        # Frequencies
        freq: dict[str, float] = Counter(encoded)
        ##################################

        pbar = tqdm(range(self.max_iters))
        for i in pbar:
            pair_freq = self.get_pair_freq(text)
            length_dict = self.num_tokens()
            cross_entropy = -sum(v / len(encoded) * np.log(v / len(encoded)) if v != 0 else 0 for v in freq.values())

            words = self.update(pair_freq=pair_freq, freq=freq)

            bayes = length_dict + cross_entropy * len(encoded)
            pbar.set_description(
                f"CE: {cross_entropy * len(encoded):.3f}, length: {length_dict}, chosen words = {words}"
            )
            # Early stopping
            if bayes > best_value:
                best_value = bayes
                best_dict = self.dictionary
            tqdm.write(f"{self.get_tokens()}, {i}")
            if (pair_freq[words] > 1
                    or words[0] not in self.alphabet
                    or words[1] not in self.alphabet):
                key = self.concat_words(words[0], words[1])
                add_branch(tree=self.dictionary, branch=key)
            # Updating the variable for the next iteration
            encoded = self.encode(text)
            freq: dict[str, float] = Counter(encoded)

        self.dictionary = best_dict

        # Removing unused token
        # n_tokens = len(self.get_tokens())
        # self.remove_unused_token(freq_encoded=freq)
        # print(f" {n_tokens - len(self.get_tokens())} tokens removed ".center(80, "#"))

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
            root = TreeNode(key="")
            for el in json_file["alphabet"]:
                root.add_child(el)

            for word in json_file["subword"]:
                add_branch(root, word)

    def remove_unused_token(self, freq_encoded: dict[str, float]):
        # these are all the tokens that have been used for encoding the text
        token_used: list[str] = list(freq_encoded.keys())
        all_tokens = self.get_tokens()
        token_unused = list(set(all_tokens) - set(token_used) - self.alphabet)
        for token in token_unused:
            if is_branch(self.dictionary, token):
                remove_branch(tree=self.dictionary, branch=token)
