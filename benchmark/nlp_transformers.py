import torch
import tempfile
import argparse
import os
from torch.utils.data import DataLoader
from torch.profiler import ExecutionGraphObserver
from torch.optim import AdamW
from torch.autograd.profiler import record_function
from datasets import ClassLabel, load_dataset
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    PretrainedConfig,
    get_scheduler,
)


def getGPT2ModelDatasetOptimizer(accelerator, args):
    # Dataset
    raw_datasets = load_dataset("conll2003")
    column_names = raw_datasets["train"].column_names
    features = raw_datasets["train"].features
    text_column_name = "tokens" if "tokens" in column_names else column_names[0]
    label_column_name = "ner_tags" if "ner_tags" in column_names else column_names[1]

    # In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
    # unique labels.
    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list

    # If the labels are of type ClassLabel, they are already integers and we have the map stored somewhere.
    # Otherwise, we have to get the list of labels manually.
    labels_are_int = isinstance(features[label_column_name].feature, ClassLabel)
    if labels_are_int:
        label_list = features[label_column_name].feature.names
        label_to_id = {i: i for i in range(len(label_list))}
    else:
        label_list = get_label_list(raw_datasets["train"][label_column_name])
        label_to_id = {l: i for i, l in enumerate(label_list)}
    num_labels = len(label_list)

    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True, add_prefix_space=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Model
    config = AutoConfig.from_pretrained("gpt2", num_labels=num_labels)
    model = AutoModelForTokenClassification.from_pretrained(
        "gpt2", config=config,
        ignore_mismatched_sizes=False,
    )
    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Model has labels -> use them.
    if model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id:
        if sorted(model.config.label2id.keys()) == sorted(label_list):
            # Reorganize `label_list` to match the ordering of the model.
            if labels_are_int:
                label_to_id = {i: int(model.config.label2id[l]) for i, l in enumerate(label_list)}
                label_list = [model.config.id2label[i] for i in range(num_labels)]
            else:
                label_list = [model.config.id2label[i] for i in range(num_labels)]
                label_to_id = {l: i for i, l in enumerate(label_list)}
        else:
            print(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {sorted(model.config.label2id.keys())}, dataset labels:"
                f" {sorted(label_list)}.\nIgnoring the model labels as a result.",
            )

    # Set the correspondences label/ID inside the model config
    model.config.label2id = {l: i for i, l in enumerate(label_list)}
    model.config.id2label = dict(enumerate(label_list))

    # Tokenize all texts and align the labels with them.
    def tokenize_function(examples):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            padding="max_length",
            truncation=True,
            max_length=128,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )
        labels = []
        for i, label in enumerate(examples[label_column_name]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[label[word_idx]])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx

            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    with accelerator.main_process_first():
        processed_raw_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names
        )
    train_dataset = processed_raw_datasets["train"]

    # DataLoader
    data_collator = DataCollatorForTokenClassification(
        tokenizer, pad_to_multiple_of=(8 if accelerator.mixed_precision == "fp16" else None)
    )
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.batch_size
    )

    # Optimizer: Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5)

    return model, train_dataloader, optimizer


def getBERTModelDatasetOptimizer(accelerator, args):
    # Model
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)

    # Dataset
    raw_datasets = load_dataset("yelp_review_full")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True
        )
    with accelerator.main_process_first():
        processed_raw_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"]
        )
        processed_raw_datasets = processed_raw_datasets.rename_column("label", "labels")
        processed_raw_datasets.set_format("torch")
    train_dataset = processed_raw_datasets["train"]

    # DataLoader
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=args.batch_size
    )

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5)

    return model, train_dataloader, optimizer


MODEL_DATASET_OPTIMIZER = {
    "gpt2": getGPT2ModelDatasetOptimizer,
    "bert": getBERTModelDatasetOptimizer,
}


def main():
    parser = argparse.ArgumentParser(description="NLP benchmark")
    parser.add_argument('--enable-profiling', action='store_true', default=False,
                       help='enable autograd profiler')
    parser.add_argument("--profile-out-dir", type=str, default="profiles/tmp")
    parser.add_argument('--model-name',  action='store', default='bert',
                       choices=['bert', 'gpt2'],
                       help='model name can be specified. bert is default.')
    parser.add_argument('--collect-execution-graph', action='store_true', default=False,
                       help='collect execution graph')
    parser.add_argument("--batch-size", type=int, default=8,
                       help='batch size')
    parser.add_argument("--gradient-accumulation-steps", type=int, default=5,
                       help='gradient accumulation steps')
    parser.add_argument("--num-warmup-iters", type=int, default=5,
                       help='number of warmup iterations')
    parser.add_argument("--num-batches", type=int, default=50,
                       help='number of batches in loop to average perf')
    parser.add_argument("--print-freq", type=int, default=5,
                       help='print frequency')
    args = parser.parse_args()
    accelerator = Accelerator()

    # Set seed before initializing model.
    set_seed(42)
    accelerator.wait_for_everyone()

    # Get model and data and optimizer
    func = MODEL_DATASET_OPTIMIZER[args.model_name]
    model, train_dataloader, optimizer = func(accelerator, args)

    # Learning rate scheduler
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_iters,
        num_training_steps=args.num_batches,
    )

    # Use the device given by the `accelerator` object.
    device = accelerator.device
    model.to(device)

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # EG
    fp = tempfile.NamedTemporaryFile('w+t', prefix='/tmp/pytorch_execution_graph_', suffix='.json', delete=False)
    fp.close()
    eg = ExecutionGraphObserver()
    eg.register_callback(fp.name)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    completed_steps = 0
    total_time = 0
    model.train()
    with torch.autograd.profiler.profile(
        args.enable_profiling, use_cuda=True, use_kineto=True, record_shapes=False
    ) as prof:
        while True:
            if completed_steps >= args.num_warmup_iters + args.num_batches:
                break
            for step, batch in enumerate(train_dataloader):
                if completed_steps >= args.num_warmup_iters + args.num_batches:
                    break

                if completed_steps == 0:
                    eg.start()
                should_print = ((completed_steps + 1) % args.print_freq) == 0

                start_event.record()
                with record_function("## Forward ##"):
                    outputs = model(**batch)

                with record_function("## Backward ##"):
                    loss = outputs.loss
                    loss = loss / args.gradient_accumulation_steps
                    accelerator.backward(loss)

                    # Gradient accumulation to simulate big batch
                    if step % args.gradient_accumulation_steps == 0 or \
                            step == len(train_dataloader) - 1:
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()
                end_event.record()
                torch.cuda.synchronize()

                if completed_steps == 0:
                    eg.stop()
                    eg.unregister_callback()

                completed_steps += 1
                if completed_steps > args.num_warmup_iters:
                    total_time += start_event.elapsed_time(end_event)
                    if should_print:
                        time_per_it = total_time / (completed_steps - args.num_warmup_iters) # ms
                        print("Finished step {}/{}, {:.2f} ms/it".format(
                            completed_steps - args.num_warmup_iters, args.num_batches, time_per_it
                        ))

    print("Overall per-batch training time: {:.2f} ms".format(
        total_time / (completed_steps - args.num_warmup_iters)
    )) # ms

    if args.enable_profiling:
        if accelerator.num_processes > 1: # Multiple trace files for distributed training
            with open("nlp_{}.prof".format(accelerator.local_process_index), "w") as prof_f:
                prof_f.write(prof.key_averages().table(sort_by="self_cpu_time_total"))
            prof.export_chrome_trace(os.path.join(args.profile_out_dir, "nlp_{}.json".format(accelerator.local_process_index)))
        else:
            with open("nlp.prof", "w") as prof_f:
                prof_f.write(prof.key_averages().table(sort_by="self_cpu_time_total"))
            prof.export_chrome_trace(os.path.join(args.profile_out_dir, "nlp.json"))


if __name__ == "__main__":
    main()