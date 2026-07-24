from __future__ import annotations

import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from itertools import repeat
from pathlib import Path
from collections.abc import  Iterator, Sequence
from typing import BinaryIO

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