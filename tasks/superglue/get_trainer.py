import logging
import os
import random
import sys
import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelWithHeads,
    AdapterConfig
)

from model.utils import get_model, TaskType

from training.trainer_base import BaseTrainer, BaseAdapterTrainer
from transformers import  Trainer, AdapterTrainer, EarlyStoppingCallback, set_seed
from tasks.superglue.dataset import SuperGlueDataset
# from training.trainer import Trainer

logger = logging.getLogger(__name__)

def get_trainer(args):
    model_args, data_args, training_args, _, adapter_args = args
    print("set model randome seed ", model_args.model_seed)
    set_seed(model_args.model_seed)
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)

    
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
    )
    # if data_args.dataset_name == 'record':
    #     dataset = SuperGlueDatasetForRecord(tokenizer, data_args, training_args)
    # else:
    dataset = SuperGlueDataset(tokenizer, data_args, training_args)

    # if training_args.do_train:
    #     for index in random.sample(range(len(dataset.train_dataset)), 3):
    #         logger.info(f"Sample {index} of the training set: {dataset.train_dataset[index]}.")
    if not dataset.multiple_choice:
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
    if not dataset.multiple_choice:
        model = get_model(model_args, TaskType.SEQUENCE_CLASSIFICATION, config)
    else:
        model = get_model(model_args, TaskType.MULTIPLE_CHOICE, config, fix_bert=True)


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
    trained_param = 0
    
    for n, p in param_optimizer:
        if p.requires_grad:
            trained_param += p.numel()
            logger.info(f"{n}")

    set_seed(training_args.seed)
    print("set data randome seed ", training_args.seed)
    trainer_cls = AdapterTrainer if adapter_args.train_adapter else Trainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=dataset.train_dataset if training_args.do_train else None,
        eval_dataset=dataset.eval_dataset if training_args.do_eval else None,
        compute_metrics=dataset.compute_metrics,
        tokenizer=tokenizer,
        data_collator=dataset.data_collator,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=model_args.patient)]
    )


    return trainer, dataset.predict_dataset
