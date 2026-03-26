from typing import Optional

from tokenizer.tokenizer import Tokenizer
from utils.tree import add_branch, is_branch


class WordPiece(Tokenizer):
    def __init__(
            self,
            alphabet: set[str],
            max_iters: int = 100,
            max_length: int = 6,
    ):
        super().__init__(alphabet=alphabet, max_iters=max_iters, max_length=max_length)
        for char in alphabet:
            add_branch(self.dictionary, "##" + char)

    def encode_word(self, word: str) -> list[str]:
        """
        This function encodes a word into a list of tokens in the form of wordpiece encoding.

        :param word:
        :param vocab:
        :return:
        """
        tokens = []
        i = 0
        n = len(word)

        while i < n:
            match = None

            # Try longest possible substring
            for j in range(n, i, -1):
                prefix = "" if i == 0 else "##"
                candidate = prefix + word[i:j]

                # Check whether a word is inside the vocabulary or not
                if is_branch(self.dictionary, prefix + word[i:j]):
                    match = candidate
                    i = j
                    break

            if match is None:
                return ["[UNK]"]

            tokens.append(match)
        return tokens

    def encode(self, text: str) -> list[str]:
        tokens = []

        # Basic whitespace tokenization (can be replaced)
        basic_tokens = self.basic_tokenizer(text)
        for token in basic_tokens:
            if token in {"\n", "\t", " "}:
                tokens.append(token)
            else:
                tokens.extend(self.encode_word(token))
        return tokens

    def concat_words(self, word_a: str, word_b: str) -> str:
        if word_b.startswith("##"):
            return word_a + word_b[2:]
        return word_a + word_b

    def are_concatenable(self, word_a: str, word_b: str) -> bool:
        """
        Check if two tokens can be concatenated.
        In WordPiece, only continuation tokens (starting with ##) can follow.

        Args:
            word_a: First token
            word_b: Second token

        Returns:
            True if word_b is a continuation token
        """
        return word_b.startswith("##")

    def check_vocabulary(self, text: str):
        super().check_vocabulary(text)

        for char in self.alphabet:
            add_branch(self.dictionary, "##" + char)
        print(self.get_tokens())

    def update(
            self,
            pair_freq: Optional[dict[tuple[str, str], float]] = None,
            freq: Optional[dict[str, float]] = None
    ) -> tuple[str, str]:
        best_pair = None
        best_score = -float("inf")

        for (token1, token2), pair_count in pair_freq.items():
            if pair_count == 0:
                continue

            token1_count = freq.get(token1, 0)
            token2_count = freq.get(token2, 0)

            if token1_count == 0 or token2_count == 0:
                continue

            # WordPiece likelihood score
            score = pair_count / (token1_count * token2_count)
            if score > best_score and self.are_concatenable(token1, token2):
                best_score = score
                best_pair = (token1, token2)

        # Return the pair with highest score
        return best_pair

    def decode(self, encoded_text: list[str]) -> str:
        result = []
        for token in encoded_text:
            if token == "[UNK]":
                result.append("[UNK]")
            elif token.startswith("##"):
                # Continuation token - append without space
                if result:
                    result[-1] += token[2:]
                else:
                    # Edge case: starts with continuation token
                    result.append(token[2:])
            else:
                # New word token
                result.append(token)

        return " ".join(result)
