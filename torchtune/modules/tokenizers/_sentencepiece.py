# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

from sentencepiece import SentencePieceProcessor
from torchtune.modules.tokenizers._utils import BaseTokenizer


# start and end header tokens for formatting chat messages
START_HEADER_ID = "<start_of_turn>"
END_HEADER_ID = "\n"
EOT_ID = "<end_of_turn>"

PAD_ID = -1


WHITESPACE_CHARS = [" ", "\n", "\t", "\r", "\v"]


class SentencePieceBaseTokenizer(BaseTokenizer):
    """
    A light-weight wrapper around SentencePieceProcessor that additionally handles
    trimming leading whitespaces.

    Args:
        path (str): Path to pretrained tokenizer file.

    Examples:
        >>> tokenizer = SentencePieceBaseTokenizer("/path/to/spm_model")
        >>> tokenized_text = tokenizer.encode("Hello world!", add_bos=True, add_eos=True)
        >>> print(tokenized_text)
        [1, 31587, 29644, 102, 2]
    """

    def __init__(
        self,
        path: str,
        start_header_id: str = START_HEADER_ID,
        end_header_id :str = END_HEADER_ID,
        eot_id: str = EOT_ID,
    ):
        spm_model = SentencePieceProcessor()
        spm_model.load(path)
        self.spm_model = spm_model
        self.vocab_size = spm_model.vocab_size()
        self.bos_id = spm_model.bos_id()
        self.eos_id = spm_model.eos_id()
        self.pad_id = spm_model.pad_id()

        # on whitespace characters and encode them separately.
        self.encodes_whitespace = any(
            [self.spm_model.encode(c) for c in WHITESPACE_CHARS]
        )

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
        trim_leading_whitespace: bool = False,
        prefix: Optional[str] = None,
    ) -> List[int]:
        """Encode text into token IDs.

        Args:
            text (str): The input text to be encoded, unbatched.
            add_bos (bool): Whether to prepend BOS to the input, defaults to True.
            add_eos (bool): Whether to append EOS to the input, defaults to True.
            trim_leading_whitespace (bool): Whether to trim leading whitespace from
                underlying sentencepiece tokenization. Sentencepiece normally prepends
                whitespace to any tokenized text, which can cause differences where
                encode(s1) + encode(s2) != encode(s1 + s2) due to leading whitespace
                added to s2. This will only trim leading whitespace if the underlying
                ``SentencePieceProcessor`` encodes whitespace. Default: False
            prefix (Optional[str]): Optional string to encode for trimming leading
                whitespaces. Used only if trim_leading_whitespace=True. Default: None

        Returns:
            List[int]: The encoded token IDs.
        """
        # We typically trim leading whitespace on the next message when
        # it is a continuation of the turn (i.e. not the first message)
        # or the previous message did not end with a space. This is handled
        # by the caller of this method. We only need to trim leading whitespace
        # if the underlying SentencePieceProcessor encodes whitespace.
        if trim_leading_whitespace and self.encodes_whitespace:
            # Can define our own custom prefix depending on vocab if needed
            if not hasattr(self, "prefix"):
                self.prefix = prefix or "\n"
                self.encoded_prefix = self.spm_model.encode(
                    self.prefix, add_bos=False, add_eos=False
                )
            start_idx = len(self.encoded_prefix) + int(add_bos)
            return self.spm_model.encode(
                self.prefix + text,
                add_bos=add_bos,
                add_eos=add_eos,
                out_type=int,
            )[start_idx:]
        else:
            return self.spm_model.encode(
                text,
                add_bos=add_bos,
                add_eos=add_eos,
                out_type=int,
            )

    def encode_with_special_tokens(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
        trim_leading_whitespace: bool = False,
        prefix: Optional[str] = None,
    ) -> List[int]:

        return self.encode(
            text=text,
            add_bos=add_bos,
            add_eos=add_eos,
            trim_leading_whitespace=trim_leading_whitespace,
            prefix=prefix,
        )

    def decode(self, ids: List[int], truncate_at_eos: bool = True) -> str:
        """Decode token IDs to strings.

        Args:
            ids (List[int]): The input token IDs to be decoded.

        Returns:
            str: The decoded text.
        """
        if truncate_at_eos:
            try:
                k = ids.index(self.eos_id)
            except ValueError:
                k = None
            if k:
                ids = ids[:k]

        return self.spm_model.decode(ids)