from enum import Enum


from model.sequence_classification import (
    RobertaPrefixForSequenceClassification,
    RobertaLoraForSequenceClassification
)


from model.multiple_choice import (
    RobertaPrefixForMultipleChoice,
    RobertaLoraForMultipleChoice
)

from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoModelForMultipleChoice
)



class TaskType(Enum):
    TOKEN_CLASSIFICATION = 1,
    SEQUENCE_CLASSIFICATION = 2,
    QUESTION_ANSWERING = 3,
    MULTIPLE_CHOICE = 4

PREFIX_MODELS = {

    "roberta": {
        TaskType.SEQUENCE_CLASSIFICATION: RobertaPrefixForSequenceClassification,
        TaskType.MULTIPLE_CHOICE: RobertaPrefixForMultipleChoice,
        
    },
}



AUTO_MODELS = {
    TaskType.TOKEN_CLASSIFICATION: AutoModelForTokenClassification,
    TaskType.SEQUENCE_CLASSIFICATION: AutoModelForSequenceClassification,
    TaskType.QUESTION_ANSWERING: AutoModelForQuestionAnswering,
    TaskType.MULTIPLE_CHOICE: AutoModelForMultipleChoice,
}

LORA_MODELS = {
    "roberta": {
        TaskType.SEQUENCE_CLASSIFICATION: RobertaLoraForSequenceClassification,
        TaskType.MULTIPLE_CHOICE: RobertaLoraForMultipleChoice,
    },

}

def get_model(model_args, task_type: TaskType, config: AutoConfig, fix_bert: bool = False):
    if model_args.prefix:
        config.hidden_dropout_prob = model_args.hidden_dropout_prob
        config.pre_seq_len = model_args.pre_seq_len
        config.prefix_projection = model_args.prefix_projection
        config.prefix_hidden_size = model_args.prefix_hidden_size
        
        model_class = PREFIX_MODELS[config.model_type][task_type]
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            revision=model_args.model_revision,
        )
    elif model_args.lora:
        config.lora = model_args.lora
        config.lora_r = model_args.lora_r
        config.lora_alpha = model_args.lora_alpha
        model_class = LORA_MODELS[config.model_type][task_type]
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            revision=model_args.model_revision,
        )
    else:
        model_class = AUTO_MODELS[task_type]
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            revision=model_args.model_revision,
        )

        bert_param = 0
        if fix_bert:
            if config.model_type == "bert":
                for param in model.bert.parameters():
                    param.requires_grad = False
                for _, param in model.bert.named_parameters():
                    bert_param += param.numel()
            elif config.model_type == "roberta":
                for param in model.roberta.parameters():
                    param.requires_grad = False
                for _, param in model.roberta.named_parameters():
                    bert_param += param.numel()
            elif config.model_type == "deberta":
                for param in model.deberta.parameters():
                    param.requires_grad = False
                for _, param in model.deberta.named_parameters():
                    bert_param += param.numel()
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
        total_param = all_param - bert_param
        print('***** total param is {} *****'.format(total_param))
    return model


