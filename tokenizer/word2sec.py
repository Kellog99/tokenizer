from typing import Optional

from tokenizer.tokenizer import Tokenizer


class WordPiece(Tokenizer):
    def __init__(
            self,
            alphabet: set[str],
            max_iters: int = 100,
            max_length: int = 4,
    ):
        super().__init__(alphabet=alphabet, max_iters=max_iters, max_length=max_length)

    def update(
            self,
            pair_freq: Optional[dict[tuple[str, str], float]] = None,
            freq: Optional[dict[str, float]] = None
    ) -> tuple[str, str]:
        ll = {
            words: value / (freq[words[0]] * freq[words[1]])
            for words, value in pair_freq.items()
            if value != 0 and freq[words[0]] != 0 and freq[words[1]] != 0
        }
        words, words_freq = max(ll.items(), key=lambda item: item[1])
        return words
