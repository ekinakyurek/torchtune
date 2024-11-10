# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Any, Callable, Dict, Optional

from torchtune.data._messages import ShareGPTToMessages, ARCMultiModalToMessages
from torchtune.datasets._sft import SFTDataset
from torchtune.modules.transforms import Transform


# TODO: point to Flamingo model transform as an example
def arc_multimodal_dataset(
    model_transform: Transform,
    *,
    source: str = "liuhaotian/LLaVA-Instruct-150K",
    image_dir: str = "coco/train2017/",
    column_map: Optional[Dict[str, str]] = None,
    train_on_input: bool = False,
    unmask_outputs: bool = False,
    new_system_prompt: Optional[str] = None,
    packed: bool = False,
    filter_fn: Optional[Callable] = None,
    split: str = "train",
    **load_dataset_kwargs: Dict[str, Any],
) -> SFTDataset:
    """
    .. code-block:: python

        from torchtune.models.llama3 import llama3_tokenizer
        from torchtune.models.clip import CLIPImageTransform
        from torchtune.modules.transforms import Transform

        class MyModelTransform(Transform):
            def __init__(
                self,
                tokenizer_path: str,
                max_seq_len: Optional[int] = None,
            ):
                self.tokenizer = llama3_tokenizer(tokenizer_path)
                self.image_transform = CLIPImageTransform()

            def __call__(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
                tokens, mask = self.tokenizer.tokenize_messages(sample["messages"])
                images = self.image_transform(sample["images"])
                return {
                    "tokens": tokens,
                    "mask": mask,
                    "images": images,
                }

    See :class:`~torchtune.datasets.SFTDataset` for more details about model transforms and
    message transforms.

    Args:
        model_transform (Transform): model-specific transform class that takes in a sample dict and applies custom
            transforms on the keys. It should consist of at minimum two components: text tokenization (called
            on the "messages" field) and image transform (called on the "images" field). The keys returned by
            the model transform should be aligned with the expected inputs into the model.
        source (str): path to dataset repository on Hugging Face. For local datasets,
            define source as the data file type (e.g. "json", "csv", "text") and pass
            in the filepath in ``data_files``. See `Hugging Face's
            <https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path>`_
        image_dir (str): path to the directory containing the images as you are expected to download the COCO dataset
            before using. Default is "coco/".
        column_map (Optional[Dict[str, str]]): a mapping from the expected columns ("conversations")
            to the new column names in the dataset. If None, assume these are identical.
            Default is None.
        new_system_prompt (Optional[str]): if specified, prepend a system message. This can
            serve as instructions to guide the model response. Setting this will OVERRIDE any system
            messages already present in the dataset. Default is None.
        packed (bool): Whether or not to pack the dataset to ``max_seq_len`` prior to training. Default is False.
        filter_fn (Optional[Callable]): callable used to filter the dataset prior to any pre-processing. See
            the Hugging Face `docs <https://huggingface.co/docs/datasets/v2.20.0/process#select-and-filter>`_ for more
            details.
        split (str): ``split`` argument for ``datasets.load_dataset``. You can use this argument to load a subset
            of a given split, e.g. ``split="train[:10%]"``. Default is "train".
        data_files (str): path to the json file to load as dataset. See the `dataset repo
            <https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/tree/main>`_ for options.
            Default is "llava_instruct_150k.json".
        **load_dataset_kwargs (Dict[str, Any]): additional keyword arguments to pass to ``load_dataset``. See Hugging
            Face's `API ref <https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset>`_
            for more details.

    Returns:
        SFTDataset: dataset configured with source data and transform

    Raises:
        ValueError: If ``packed`` is True, they are not supported for multimodal datasets yet.

    Example:
        >>> llava_instruct_ds = llava_instruct_dataset(model_transform=model_transform)
        >>> for batch in Dataloader(llava_instruct_ds, batch_size=8):
        >>>     print(f"Batch size: {len(batch)}")
        >>> Batch size: 8
    """

    message_transform = ARCMultiModalToMessages(
        train_on_input=False,
        column_map=column_map,
    )


    ds = SFTDataset(
        model_transform=model_transform,
        source=source,
        message_transform=message_transform,
        filter_fn=filter_fn,
        split=split,
        data_files = {"train": "td_False_ttd_False_ttdwa_False_ad_True_trd_False.jsonl",
                      "test": "td_True_ttd_False_ttdwa_False_ad_True_trd_False.jsonl",},
        **load_dataset_kwargs,
    )
    if packed:
        raise ValueError("Multimodal datasets don't support packing yet.")
    return ds
