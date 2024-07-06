from datasets import load_dataset, DatasetDict, Dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer
)
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig
import evaluate
import torch
import numpy as np

model_checkpoint = 'distilbert-base-uncased'

id2label = {0: "Negative", 1: "Positive"}
label2id = {"Negative": 0, "Positive": 1}

model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels = 2, id2label = id2label,  label2id = label2id)
dataset = load_dataset("shawhin/imdb-truncated")

# dataset = 
# DatasetDict({
    #     'train': Dataset({
    #         features: ['text', 'label'],
    #         num_rows: 1000})
    #     'test': Dataset({
    #         features: ['text', 'label'],
    #         num_rows: 1000})})

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space = True)

def tokenize_function(examples):
    text = examples['text']
    tokenizer.truncation_side = "left"
    tokenized_inputs = tokenizer(
        text,
        return_tensors = "np",
        truncation = True,
        max_length = 512
    )
    return tokenized_inputs

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_tokenize_embeddings(len(tokenizer))

tokenized_datasets = dataset.map(tokenize_function, batched = True)

# tokenized_dataset = 
# DatasetDict({
    #     'train': Dataset({
    #         features: ['text', 'label', 'attention_mask'],
    #         num_rows: 1000})
    #     'validation': Dataset({
    #         features: ['text', 'label', 'attention_mask'],
    #         num_rows: 1000})})

data_collator = DataCollatorWithPadding(tokenizer = tokenizer)

accuracy = evaluate.load("accuracy")

def computer_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis = 1)
    return {"accuracy": accuracy.compute(predictions = predictions, references = labels)["accuracy"]}

text_list = ["It was good.", "Not a fan, don't recommend",
             "I hated it, it was terrible", "I loved it, it was amazing",
             "It was okay, not the best", "It was bad, I don't recommend it"]

print("Untrained model predictions:")
print("----------------------------")
for text in text_list:
    inputs = tokenizer.encode(text, return_tensors = "pt")
    logits = model(inputs).logits
    predictions = torch.argmax(logits)
    print(text + " - " + id2label[predictions.tolist()])

peft_config = LoraConfig(task_type = "SEQ_CLS",     # sequence classification
                         r = 4,                     # intrinsic rank of trainable weight matrix
                         lora_alpha = 32,           # like learning rate
                         lora_dropout = 0.01,       # probability of dropout
                         target_modules = ['q_lin'])# apply lora to query layer

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# hyperparameters
lr = 1e-3
batch_size = 4
num_epochs = 4

training_args = TrainingArguments(
    output_dir = model_checkpoint + "-lora-text-classification",
    num_train_epochs = num_epochs,
    learning_rate = lr,
    per_device_train_batch_size = batch_size,
    per_device_eval_batch_size = batch_size,
    weight_decay = 0.01,
    eval_strategy = "epoch",
    save_strategy = "epoch",
    load_best_model_at_end = True
)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = tokenized_datasets["train"],
    eval_dataset = tokenized_datasets["validation"],
    data_collator = data_collator,
    compute_metrics = computer_metrics,
    tokenizer = tokenizer
)

trainer.train()

model.to("cpu")

print("Trained model predictions:")
print("--------------------------")
for text in text_list:
    inputs = tokenizer.encode(text, return_tensors = "pt").to("cpu")
    logits = model(inputs).logits
    predictions = torch.max(logits, 1).indices
    print(text + " - " + id2label[predictions.tolist()[0]])