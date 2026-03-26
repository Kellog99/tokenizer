from typing import Optional

from tqdm import tqdm

from tokenizer.tokenizer import Tokenizer
from utils.tree import TreeNode


class BytePairEncoding(Tokenizer):
    def __init__(
            self,
            alphabet: Optional[set[str]] = None,
            max_iters: int = 100,
            max_length: int = 6,
    ):
        super().__init__(alphabet=alphabet, max_iters=max_iters, max_length=max_length)

    def update(
            self,
            pair_freq: Optional[dict[tuple[str, str], float]] = None,
            freq: Optional[dict[str, float]] = None
    ) -> tuple[str, str]:
        words, max_freq = max(pair_freq.items(), key=lambda x: x[1])
        tqdm.write(f"{words} = {max_freq}")

        return words

    def encode(self, text: str) -> list[str]:
        out = []
        i = 0
        while i < len(text):
            node: TreeNode = self.dictionary
            word = ""

            while i < len(text) and node.is_child(text[i]):
                word += text[i]
                node = node.children[text[i]]
                i += 1

            if len(word) > 0:
                out.append(word)

        return out
