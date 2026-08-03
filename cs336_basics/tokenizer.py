from __future__ import annotations

import os
from collections import Counter, defaultdict
from dataclasses import dataclass

from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from itertools import repeat
from pathlib import Path
from collections.abc import  Iterator, Sequence, Iterable
from typing import BinaryIO
import json
import regex as re


PRETOKEN_PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)|"""
    r""" ?\p{L}+|"""
    r""" ?\p{N}+|"""
    r""" ?[^\s\p{L}\p{N}]+|"""
    r"""\s+(?!\S)|"""
    r"""\s+"""
)


def validate_special_tokens(
    special_tokens: Sequence[bytes],
) -> tuple[bytes, ...]:
    """
    Validate and deduplicate special tokens while preserving order.

    An empty sequence is allowed.
    """
    normalized = tuple(dict.fromkeys(special_tokens))

    for token in normalized:
        if not isinstance(token, bytes):
            raise TypeError(
                "Every special token must be a bytes object"
            )

        if not token:
            raise ValueError(
                "Special tokens cannot be empty"
            )

        try:
            token.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(
                f"Special token is not valid UTF-8: {token!r}"
            ) from error

    return normalized


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    special_tokens: Sequence[bytes],
) -> list[int]:
    """
    Find safe worker boundaries at the beginning of special tokens.

    This function does not remove special tokens. It only ensures that a
    special token is not divided between two worker chunks. Actual removal
    occurs in `pretokenize_chunk`.
    """
    if desired_num_chunks < 1:
        raise ValueError(
            "desired_num_chunks must be at least 1"
        )

    tokens = validate_special_tokens(special_tokens)

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    if file_size == 0:
        return [0]

    # Without special tokens, arbitrary byte boundaries could split a UTF-8
    # character or pretoken. Process the corpus as one chunk instead.
    if desired_num_chunks == 1 or not tokens:
        return [0, file_size]

    chunk_size = file_size // desired_num_chunks

    boundaries = [
        index * chunk_size
        for index in range(desired_num_chunks + 1)
    ]
    boundaries[-1] = file_size

    max_token_length = max(
        len(token)
        for token in tokens
    )

    read_size = max(4096, max_token_length)

    # Overlap reads so tokens crossing a read-window boundary are detected.
    step_size = max(
        1,
        read_size - max_token_length + 1,
    )

    for boundary_index in range(
        1,
        len(boundaries) - 1,
    ):
        search_position = boundaries[boundary_index]

        while search_position < file_size:
            file.seek(search_position)
            block = file.read(read_size)

            if not block:
                boundaries[boundary_index] = file_size
                break

            matching_offsets = [
                offset
                for token in tokens
                if (offset := block.find(token)) != -1
            ]

            if matching_offsets:
                # Place the boundary immediately before the earliest
                # special token found after the guessed boundary.
                boundaries[boundary_index] = (
                    search_position + min(matching_offsets)
                )
                break

            if len(block) < read_size:
                boundaries[boundary_index] = file_size
                break

            search_position += step_size

    return sorted(set(boundaries))


def generate_chunks(
    file: BinaryIO,
    boundaries: Sequence[int],
) -> Iterator[str]:
    """
    Yield every contiguous file range.

    Special tokens may still occur in these strings. They are removed later
    by `pretokenize_chunk`.
    """
    for start, end in zip(
        boundaries[:-1],
        boundaries[1:],
    ):
        file.seek(start)
        chunk_bytes = file.read(end - start)

        yield chunk_bytes.decode("utf-8")


@lru_cache(maxsize=32)
def compile_special_token_pattern(
    special_tokens: tuple[str, ...],
) -> re.Pattern | None:
    """
    Compile a pattern matching every special token.

    Longer tokens are matched first to handle overlapping tokens correctly.
    """
    if not special_tokens:
        return None

    ordered_tokens = sorted(
        set(special_tokens),
        key=len,
        reverse=True,
    )

    expression = "|".join(
        re.escape(token)
        for token in ordered_tokens
    )

    return re.compile(expression)


def pretokenize_chunk(
    chunk: str,
    special_tokens: tuple[str, ...],
) -> Counter[bytes]:
    """
    Remove all special tokens, then pretokenize the remaining text.

    Every occurrence of every special token is omitted, regardless of how
    many worker boundaries were created.
    """
    counts: Counter[bytes] = Counter()

    special_token_pattern = compile_special_token_pattern(
        special_tokens
    )

    if special_token_pattern is None:
        sections = (chunk,)
    else:
        # re.split removes the matched delimiters because the pattern has
        # no capturing group.
        sections = special_token_pattern.split(chunk)

    for section in sections:
        for match in PRETOKEN_PATTERN.finditer(section):
            pretoken = match.group().encode("utf-8")
            counts[pretoken] += 1

    return counts

def pretokenize_text(text: str) -> Iterator[bytes]:
    """
    Pretokenize normal text without handling special tokens.
    """
    for match in PRETOKEN_PATTERN.finditer(text):
        yield match.group().encode("utf-8")


def pretokenize_ordered(
    text: str,
    special_tokens: Sequence[str] = (),
) -> list[bytes]:
    """
    Pretokenize text in order while preserving special tokens as atomic
    byte strings.
    """
    normalized_tokens = tuple(dict.fromkeys(special_tokens))
    special_token_pattern = compile_special_token_pattern(
        normalized_tokens
    )

    if special_token_pattern is None:
        return list(pretokenize_text(text))

    pretokens: list[bytes] = []
    previous_end = 0

    for special_match in special_token_pattern.finditer(text):
        # Pretokenize the ordinary text before this special token.
        ordinary_text = text[
            previous_end:special_match.start()
        ]
        pretokens.extend(pretokenize_text(ordinary_text))

        # Preserve the special token as one atomic pretoken.
        pretokens.append(
            special_match.group().encode("utf-8")
        )

        previous_end = special_match.end()

    # Pretokenize any ordinary text after the final special token.
    pretokens.extend(
        pretokenize_text(text[previous_end:])
    )

    return pretokens

def pretokenize_file(
    input_path: str | os.PathLike,
    special_tokens: Sequence[bytes],
    num_processes: int = 1,
) -> Counter[bytes]:
    """
    Pretokenize a UTF-8 corpus while excluding special tokens.

    The special tokens are used for safe worker boundaries and are also
    removed from every chunk before normal pretokenization.
    """
    if num_processes < 1:
        raise ValueError(
            "num_processes must be at least 1"
        )

    normalized_tokens = validate_special_tokens(
        special_tokens
    )

    special_tokens_text = tuple(
        token.decode("utf-8")
        for token in normalized_tokens
    )

    total_counts: Counter[bytes] = Counter()
    path = Path(input_path)

    with path.open("rb") as file:
        boundaries = find_chunk_boundaries(
            file=file,
            desired_num_chunks=num_processes,
            special_tokens=normalized_tokens,
        )

        chunks = generate_chunks(
            file=file,
            boundaries=boundaries,
        )

        chunk_count = max(
            0,
            len(boundaries) - 1,
        )

        if num_processes == 1 or chunk_count <= 1:
            for chunk in chunks:
                total_counts.update(
                    pretokenize_chunk(
                        chunk,
                        special_tokens_text,
                    )
                )

            return total_counts

        worker_count = min(
            num_processes,
            chunk_count,
        )

        with ProcessPoolExecutor(
            max_workers=worker_count
        ) as executor:
            results = executor.map(
                pretokenize_chunk,
                chunks,
                repeat(special_tokens_text),
            )

            for chunk_counts in results:
                total_counts.update(chunk_counts)

    return total_counts

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






Merge = tuple[bytes, bytes]


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[Merge],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = vocab
        self.merges = merges

        self.special_tokens = special_tokens or []
        self.special_token_bytes = {
            token.encode("utf-8")
            for token in self.special_tokens
        }

        # Encoding needs bytes -> token ID.
        self.token_to_id = {
            token_bytes: token_id
            for token_id, token_bytes in vocab.items()
        }

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        merges: list[Merge] = []

        with open(merges_filepath) as merges_file:
            for line in merges_file:
                tok1_hex, tok2_hex = line.split()
                merges.append(
                    (
                        bytes.fromhex(tok1_hex),
                        bytes.fromhex(tok2_hex),
                    )
                )

        with open(vocab_filepath) as vocab_file:
            hex_vocab: dict[str, int] = json.load(vocab_file)

        vocab = {
            token_id: bytes.fromhex(token_hex)
            for token_hex, token_id in hex_vocab.items()
        }

        return cls(vocab, merges, special_tokens)

    def _encode_pretoken(self, pretoken: bytes) -> list[int]:
        # Special tokens must remain atomic.
        if pretoken in self.special_token_bytes:
            return [self.token_to_id[pretoken]]

        # Iterating over bytes produces integers, so convert each byte
        # back into a one-byte bytes object.
        tokens = [bytes([byte_value]) for byte_value in pretoken]

        # Merges are already ordered by merge rank.
        for left, right in self.merges:
            merged = left + right
            index = 0

            while index < len(tokens) - 1:
                if tokens[index] == left and tokens[index + 1] == right:
                    tokens[index : index + 2] = [merged]
                    index += 1
                else:
                    index += 1

        return [self.token_to_id[token] for token in tokens]

    def encode(self, text: str) -> list[int]:
        pretokens = pretokenize_ordered(text, self.special_tokens)

        token_ids: list[int] = []

        for pretoken in pretokens:
            # Remove this conversion if pretokenize_ordered already
            # returns bytes.
            if isinstance(pretoken, str):
                pretoken = pretoken.encode("utf-8")

            token_ids.extend(self._encode_pretoken(pretoken))

        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        byte_string = b"".join(self.vocab[token_id] for token_id in ids)
        return byte_string.decode("utf-8", errors="replace")