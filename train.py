from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    PreTrainedTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperConfig,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
)


SPECIAL_TOKENS = {
    "pad": "<PAD>",
    "bos": "<s>",
    "eos": "</s>",
    "unk": "<unk>",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Streamed LibriSpeech smoke/train test for Whisper with a custom tokenizer")
    parser.add_argument("--tokenizer_dir", default="conformer_rnnt_tokenizer")
    parser.add_argument("--dataset", default="hypaai/Hypa-LibreSpeech")
    parser.add_argument("--config", default="default")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output_dir", default="whisper-custom-tokenizer-output")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--max_label_length", type=int, default=448)
    parser.add_argument("--max_audio_seconds", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


@dataclass
class WhisperDataCollator:
    feature_extractor: WhisperFeatureExtractor
    tokenizer: PreTrainedTokenizerFast
    pad_token_id: int

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        inputs = [feature["input_features"] for feature in features]

        input_features = self.feature_extractor.pad(
            {"input_features": inputs},
            padding=True,
            return_tensors="pt",
        ).input_features

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.tokenizer.pad(
            label_features,
            padding=True,
            return_tensors="pt",
        ).input_ids

        labels_batch[labels_batch == self.pad_token_id] = -100

        return {
            "input_features": input_features,
            "labels": labels_batch,
        }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    tokenizer_path = Path(args.tokenizer_dir).resolve()
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

    vocab_size = len(tokenizer.get_vocab())
    special_ids = {
        name: tokenizer.convert_tokens_to_ids(token)
        for name, token in SPECIAL_TOKENS.items()
    }

    if any(token_id < 0 for token_id in special_ids.values()):
        raise ValueError(f"Missing one or more required special tokens: {SPECIAL_TOKENS}")
    if vocab_size != len(tokenizer):
        raise ValueError("Tokenizer reports inconsistent vocabulary sizes")

    print(f"Loading custom tokenizer from: {tokenizer_path}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Special token IDs: {special_ids}")

    config = WhisperConfig.from_pretrained("openai/whisper-tiny")
    config.vocab_size = vocab_size
    config.pad_token_id = special_ids["pad"]
    config.bos_token_id = special_ids["bos"]
    config.eos_token_id = special_ids["eos"]
    config.decoder_start_token_id = special_ids["bos"]
    config.suppress_tokens = []
    config.begin_suppress_tokens = []
    config.forced_decoder_ids = None

    model = WhisperForConditionalGeneration(config)

    print("Initialized a random Whisper-tiny model; no pretrained weights loaded.")
    print(f"Model vocab size: {model.config.vocab_size}")
    print(
        "Decoder embedding shape:",
        tuple(model.model.decoder.embed_tokens.weight.shape),
    )

    feature_extractor = WhisperFeatureExtractor(
        feature_size=80,
        sampling_rate=16000,
        return_attention_mask=False,
        pad_token_id=special_ids["pad"],
    )

    print(
        f"Streaming dataset: {args.dataset}/{args.config}, split={args.split}"
    )
    dataset = load_dataset(
        args.dataset,
        args.config,
        split=args.split,
        streaming=True,
    )

    dataset = dataset.cast_column("audio", __import__("datasets").Audio(sampling_rate=16000))

    def prepare_example(example: Dict) -> Dict:
        audio = example["audio"]
        array = np.asarray(audio["array"], dtype=np.float32)

        if len(array) == 0:
            return {"input_features": None, "labels": []}

        max_samples = int(args.max_audio_seconds * 16000)
        if len(array) > max_samples:
            array = array[:max_samples]

        input_features = feature_extractor(
            array,
            sampling_rate=16000,
            return_attention_mask=False,
        )["input_features"][0]

        labels = tokenizer(
            example["text"].lower().strip(),
            add_special_tokens=False,
            truncation=True,
            max_length=args.max_label_length,
        )["input_ids"]

        return {"input_features": input_features, "labels": labels}

    dataset = dataset.map(prepare_example, remove_columns=dataset.column_names)

    data_collator = WhisperDataCollator(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        pad_token_id=special_ids["pad"],
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        max_steps=args.steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        predict_with_generate=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to=[],
        seed=args.seed,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    print("Starting streamed training smoke test...")
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
