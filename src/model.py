import random
from torch.utils.data import DataLoader


class NGramModel:
    def __init__(self, seed: int = 42):
        self._seed = seed
        self._ngram_mapping = {}
        random.seed(self._seed)

    def fit(self, dataloader: DataLoader) -> None:
        for i, (x, y) in enumerate(dataloader):
            x = tuple(x.squeeze().tolist())
            y = y.squeeze().tolist()
            next_tokens_to_count_map = self._ngram_mapping.get(x, dict())
            count = next_tokens_to_count_map.get(y, 0)
            count += 1
            next_tokens_to_count_map[y] = count
            self._ngram_mapping[x] = next_tokens_to_count_map

    def predict(self, x):
        return self.samples(x, k=1)

    def samples(self, x, k=1):
        next_tokens_to_count_map = self._ngram_mapping.get(x)
        if next_tokens_to_count_map is None:
            raise ValueError(f'Model was not trained on data = {x}')
        next_possible_tokens = random.choices(
            population=list(next_tokens_to_count_map.keys()),
            weights=list(next_tokens_to_count_map.values()),
            k=k,
        )
        return next_possible_tokens

    def random_ngram(self):
        key = random.choice(list(self._ngram_mapping.keys()))
        return key
