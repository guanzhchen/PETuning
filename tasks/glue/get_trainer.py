import logging
import os
import random
import sys

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AdapterConfig
)

from model.utils import get_model, TaskType
from tasks.glue.dataset import GlueDataset
from training.trainer_base import BaseTrainer
from transformers import Trainer, AdapterTrainer, EarlyStoppingCallback

logger = logging.getLogger(__name__)

def get_trainer(args):
    model_args, data_args, training_args, _, adapter_args = args

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
    )
    dataset = GlueDataset(tokenizer, data_args, training_args)

    if not dataset.is_regression:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            num_labels=dataset.num_labels,
            label2id=dataset.label2id,
            id2label=dataset.id2label,
            finetuning_task=data_args.dataset_name,
            revision=model_args.model_revision,
        )
    else:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            num_labels=dataset.num_labels,
            finetuning_task=data_args.dataset_name,
            revision=model_args.model_revision,
        )
    config.lora = False
    model = get_model(model_args, TaskType.SEQUENCE_CLASSIFICATION, config)

    if adapter_args.train_adapter:
        logger.info(f"Reduction Factor: {adapter_args.adapter_reduction_factor}")
        task_name = data_args.task_name or "superglue"
        # check if adapter already exists, otherwise add it
        if task_name not in model.config.adapters:
            # resolve the adapter config
            adapter_config = AdapterConfig.load(
                adapter_args.adapter_config,
                non_linearity=adapter_args.adapter_non_linearity,
                reduction_factor=adapter_args.adapter_reduction_factor,
            )
            # load a pre-trained from Hub if specified
            # if adapter_args.load_adapter:
            #     model.load_adapter(
            #         adapter_args.load_adapter,
            #         config=adapter_config,
            #         load_as=task_name,
            #     )
            # # otherwise, add a fresh adapter
            # else:
            model.add_adapter(task_name, config=adapter_config)
        # Freeze all model weights except of those of this adapter
        model.train_adapter([task_name])
        # Set the adapters to be used in every forward pass
        model.set_active_adapters(task_name)
    else:
        if adapter_args.load_adapter:
            raise ValueError(
                "Adapters can only be loaded in adapters training mode."
                "Use --train_adapter to enable adapter training"
            )
    if model_args.bitfit:
        for name, param in model.named_parameters():
            if name.startswith('roberta') and "bias" not in name.lower():
                param.requires_grad = False
    param_optimizer = list(model.named_parameters())
    logger.info("Trainable parameters:")
    for n, p in param_optimizer:
        if p.requires_grad:
            logger.info(f"{n}")
            # print(n)

    trainer_cls = AdapterTrainer if adapter_args.train_adapter else Trainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=dataset.train_dataset if training_args.do_train else None,
        eval_dataset=dataset.eval_dataset if training_args.do_eval else None,
        compute_metrics=dataset.compute_metrics,
        tokenizer=tokenizer,
        data_collator=dataset.data_collator,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=10)]
    )

    return trainer, dataset.predict_dataset