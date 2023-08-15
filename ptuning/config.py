from dataclasses import dataclass
from transformers import (
    Seq2SeqTrainingArguments,
)

from arguments import ModelArguments, DataTrainingArguments


@dataclass
class PTUNINGConfig:
    base_path: str
    model: ModelArguments
    data: DataTrainingArguments
    train: Seq2SeqTrainingArguments
