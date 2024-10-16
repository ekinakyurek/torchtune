# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial

from typing import Any, Callable, Dict, Optional, Union

from torchtune.data._messages import ARCToMessages

from torchtune.datasets._packed import PackedDataset
from torchtune.datasets._sft import SFTDataset
from torchtune.modules.tokenizers import ModelTokenizer


def arc_dataset(
    tokenizer: ModelTokenizer,
    *,
    source: str = "tatsu-lab/arc",
    column_map: Optional[Dict[str, str]] = None,
    train_on_input: bool = False,
    unmask_outputs: bool = True,
    packed: bool = False,
    filter_fn: Optional[Callable] = None,
    split: str = "train",
    **load_dataset_kwargs: Dict[str, Any],
) -> Union[SFTDataset, PackedDataset]:
    """
    Masking of the prompt during training is controlled by the ``train_on_input`` flag, which is
    set to ``True`` by `default <https://github.com/tloen/arc-lora/blob/main/finetune.py#L49>`_
    - If ``train_on_input`` is True, the prompt is used during training and
    contributes to the loss.
    - If ``train_on_input`` is False, the prompt is masked out (tokens replaced with -100)

    Args:
        tokenizer (ModelTokenizer): Tokenizer used by the model that implements the ``tokenize_messages`` method.
        source (str): path to dataset repository on Hugging Face. For local datasets,
            define source as the data file type (e.g. "json", "csv", "text") and pass
            in the filepath in ``data_files``. See `Hugging Face's
            <https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path>`_
            ``load_dataset`` for more details. Default is ``tatsu-lab/arc``.
        column_map (Optional[Dict[str, str]]): a mapping from the expected columns in the message transform
            :class:`~torchtune.data.ARCToMessages` to the new column names in the dataset. Keys should be
            "instruction", "input", and "output" and values should be the actual column names. If None, uses
            the default column names ``"instruction``, ``"input"``, and ``"output"`` in ``tatsu-lab/arc``.
        train_on_input (bool): Whether the model is trained on the prompt or not. Default is False.
        packed (bool): Whether or not to pack the dataset to ``max_seq_len`` prior to training. Default is False.
        filter_fn (Optional[Callable]): callable used to filter the dataset prior to any pre-processing. See
            the Hugging Face `docs <https://huggingface.co/docs/datasets/v2.20.0/process#select-and-filter>`_ for more
            details.
        split (str): ``split`` argument for ``datasets.load_dataset``. You can use this argument to load a subset
            of a given split, e.g. ``split="train[:10%]"``. Default is "train".
        **load_dataset_kwargs (Dict[str, Any]): additional keyword arguments to pass to ``load_dataset``. See Hugging
            Face's `API ref <https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset>`_
            for more details.

    Returns:
        Union[SFTDataset, PackedDataset]: dataset configured with source data and transform

    Raises:
        ValueError: If ``packed`` is True and ``max_seq_len`` is not set on the tokenizer.

    Example:
        >>> arc_ds = arc_dataset(tokenizer=tokenizer)
        >>> for batch in Dataloader(arc_ds, batch_size=8):
        >>>     print(f"Batch size: {len(batch)}")
        >>> Batch size: 8
    """

    message_transform = ARCToMessages(
        train_on_input=train_on_input, column_map=column_map
    )
    ds = SFTDataset(
        source=source,
        message_transform=message_transform,
        model_transform=tokenizer,
        filter_fn=filter_fn,
        unmask_outputs=unmask_outputs,
        split=split,
        data_files = {"train": "td_False_ttd_False_ttdwa_False_ad_True_trd_False.jsonl",
                      "test": "td_True_ttd_False_ttdwa_False_ad_True_trd_False.jsonl",},
        **load_dataset_kwargs,
    )
    if packed:
        if tokenizer.max_seq_len is None:
            raise ValueError(
                "PackedDataset requires a max_seq_len to be set on the tokenizer."
            )
        return PackedDataset(ds, max_seq_len=tokenizer.max_seq_len)
    return ds


arc_cleaned_dataset = partial(arc_dataset, source="yahma/arc-cleaned")
arc_cleaned_dataset.__doc__ = """
Builder for a variant of ARC-style datasets with the cleaned version of the
original ARC dataset, `yahma/arc-cleaned <https://huggingface.co/datasets/yahma/arc-cleaned>`_.
See the dataset page and :func:`~torchtune.datasets.arc_dataset` for more details.
"""

if __name__ == "__main__":
    from torchtune.models.llama3 import llama3_tokenizer
    from torch.utils.data import DataLoader

    tokenizer = llama3_tokenizer("/raid/lingo//models/Meta-Llama-3-8B-Instruct/original/tokenizer.model")
    arc_ds = arc_dataset(source="/raid/lingo/akyurek/git/arc/data/tasks/all_in_pm_fix_30/", tokenizer=tokenizer)
    print(len(arc_ds))
    for batch in DataLoader(arc_ds, batch_size=8):
        print(f"Batch size: {len(batch)}")

