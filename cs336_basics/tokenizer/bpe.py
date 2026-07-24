from __future__ import annotations

import os
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from .pretokenization import pretokenize_file


Merge = tuple[bytes, bytes]
TokenPair = tuple[int, int]


@dataclass(slots=True)
class BPEModel:
    vocab: dict[int, bytes]
    merges: list[Merge]


class BPETrainer:
    BYTE_VOCAB_SIZE = 256

    def __init__(
        self,
        vocab_size: int,
        special_tokens: Sequence[bytes],
    ) -> None:
        self.special_tokens = tuple(special_tokens)
        self.vocab_size = vocab_size

    def train(
        self,
        pretoken_counts: Counter[bytes],
    ) -> BPEModel:
        vocab = self._create_initial_vocab()

        token_sequences = {
            pretoken: list(pretoken)
            for pretoken in pretoken_counts
        }

        merges: list[Merge] = []

        while len(vocab) < self.vocab_size:
            pair_frequencies, occurrences = self._count_pairs(
                pretoken_counts=pretoken_counts,
                token_sequences=token_sequences,
            )

            if not pair_frequencies:
                break

            selected_pair = self._select_pair(
                pair_frequencies=pair_frequencies,
                vocab=vocab,
            )

            left_bytes = vocab[selected_pair[0]]
            right_bytes = vocab[selected_pair[1]]

            new_token_id = len(vocab)
            vocab[new_token_id] = left_bytes + right_bytes
            merges.append((left_bytes, right_bytes))

            for pretoken in occurrences[selected_pair]:
                token_sequences[pretoken] = self._merge_pair(
                    sequence=token_sequences[pretoken],
                    pair=selected_pair,
                    new_token_id=new_token_id,
                )

        return BPEModel(
            vocab=vocab,
            merges=merges,
        )

    def _create_initial_vocab(self) -> dict[int, bytes]:
        vocab = {
            token_id: bytes([token_id])
            for token_id in range(self.BYTE_VOCAB_SIZE)
        }

        for special_token in self.special_tokens:
            vocab[len(vocab)] = special_token

        return vocab

    @staticmethod
    def _count_pairs(
        pretoken_counts: Counter[bytes],
        token_sequences: dict[bytes, list[int]],
    ) -> tuple[
        Counter[TokenPair],
        dict[TokenPair, set[bytes]],
    ]:
        frequencies: Counter[TokenPair] = Counter()
        occurrences: defaultdict[
            TokenPair,
            set[bytes],
        ] = defaultdict(set)

        for pretoken, count in pretoken_counts.items():
            sequence = token_sequences[pretoken]

            for left, right in zip(sequence, sequence[1:]):
                pair = (left, right)
                frequencies[pair] += count
                occurrences[pair].add(pretoken)

        return frequencies, dict(occurrences)

    @staticmethod
    def _merge_pair(
        sequence: list[int],
        pair: TokenPair,
        new_token_id: int,
    ) -> list[int]:
        merged: list[int] = []
        index = 0

        while index < len(sequence):
            if (
                index + 1 < len(sequence)
                and sequence[index] == pair[0]
                and sequence[index + 1] == pair[1]
            ):
                merged.append(new_token_id)
                index += 2
            else:
                merged.append(sequence[index])
                index += 1

        return merged

    @staticmethod
    def _select_pair(
        pair_frequencies: Counter[TokenPair],
        vocab: dict[int, bytes],
    ) -> TokenPair:
        """
        Choose the pair with the greatest frequency.

        Frequency ties are resolved by choosing the lexicographically
        greatest byte pair.
        """
        return max(
            pair_frequencies,
            key=lambda pair: (
                pair_frequencies[pair],
                vocab[pair[0]],
                vocab[pair[1]],
            ),
        )


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: Sequence[str],
    *,
    num_processes: int = 1,
) -> tuple[
    dict[int, bytes],
    list[tuple[bytes, bytes]],
]:
    """
    Train a byte-level BPE tokenizer.

    Special tokens are appended to the initial byte vocabulary. Their
    occurrences in the corpus are processed as ordinary text during BPE
    training.
    """
    encoded_special_tokens = tuple(
        token.encode("utf-8")
        for token in special_tokens
    )

    pretoken_counts = pretokenize_file(
        input_path=input_path,
        special_tokens=encoded_special_tokens,
        num_processes=num_processes,
    )

    trainer = BPETrainer(
        vocab_size=vocab_size,
        special_tokens=encoded_special_tokens,
    )

    model = trainer.train(pretoken_counts)

    return model.vocab, model.merges