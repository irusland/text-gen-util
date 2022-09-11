from typing import Iterable, List


class Dictionary:
    UNKNOWN = "[UNKNOWN]"
    UNKNOWN_CODE = -1

    def __init__(self):
        self._word_to_code_map = {}
        self._code_to_word_map = {}
        self._current_length = self.UNKNOWN_CODE
        self.observe(self.UNKNOWN)

    def transform(self, words: Iterable[str]) -> None:
        for word in words:
            self.observe(word)

    def observe(self, word: str) -> int:
        if not (code := self._word_to_code_map.get(word)):
            self._current_length += 1
            self._word_to_code_map[word] = self._current_length
            self._code_to_word_map[self._current_length] = word
            code = self._current_length
        return code

    def encode_many(self, words: List[str]) -> List[int]:
        return [self.encode(word) for word in words]

    def encode(self, word: str) -> int:
        return self._word_to_code_map.get(word, self.UNKNOWN_CODE)

    def decode_many(self, codes: List[int]) -> List[str]:
        return [self.decode(code) for code in codes]

    def decode(self, code: int) -> str:
        return self._code_to_word_map.get(code, self.UNKNOWN)

    def __len__(self):
        return self._current_length
