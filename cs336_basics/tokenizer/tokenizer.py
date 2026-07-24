from collections.abc import Iterable, Iterator
import json

from .pretokenization import pretokenize_ordered


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
    ) -> "Tokenizer":
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