from utils import *
from evaluate import compute_metrics

from argparse import ArgumentParser
from pathlib import Path
import numpy as np
from datasets import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer

def main(args):
    # read data
    users = read_users_data()
    group2id, id2group = read_groups_data()
    train_dataset = read_train_groups_data(users, group2id)
    validation_dataset = read_val_seen_groups_data(users, group2id)
    test_unseen_dataset = read_test_groups_data(users, 'data/test_unseen_group.csv')
    test_seen_dataset = read_test_groups_data(users, 'data/test_seen_group.csv')


    # dataset
    dataset = {}
    dataset['train'] = Dataset.from_list(train_dataset)
    dataset['validation'] = Dataset.from_list(validation_dataset)
    dataset['test_unseen'] = Dataset.from_list(test_unseen_dataset)
    dataset['test_seen'] = Dataset.from_list(test_seen_dataset)


    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")

    def preprocess_data(examples):
    # take a batch of texts
        text = examples["interests"]
        # encode them
        encoding = tokenizer(text, padding="max_length", truncation=True, max_length=128)
        # create numpy array of shape (batch_size, num_labels)
        labels = np.zeros(len(group2id))
        # fill numpy array
        for id in examples["subgroup"]:
            labels[id] = 1

        encoding["labels"] = labels.tolist()
        return encoding

    def preprocess_test_data(examples):
        # take a batch of texts
        text = examples["interests"]
        # encode them
        encoding = tokenizer(text, padding="max_length", truncation=True, max_length=128)
        return encoding

    encoded_train_dataset = dataset['train'].map(preprocess_data, remove_columns=dataset['train'].column_names)
    encoded_validation_dataset = dataset['validation'].map(preprocess_data, remove_columns=dataset['validation'].column_names)
    encoded_test_unseen_dataset = dataset['test_unseen'].map(preprocess_test_data, remove_columns=dataset['test_unseen'].column_names)
    encoded_test_seen_dataset = dataset['test_seen'].map(preprocess_test_data, remove_columns=dataset['test_seen'].column_names)


    # set format to torch
    encoded_train_dataset.set_format("torch")
    encoded_validation_dataset.set_format("torch")
    encoded_test_unseen_dataset.set_format("torch")
    encoded_test_seen_dataset.set_format("torch")


    # model
    model = AutoModelForSequenceClassification.from_pretrained("hfl/chinese-bert-wwm-ext", 
                                                            #problem_type="multi_label_classification",
                                                            #ignore_mismatched_sizes=True,
                                                            num_labels=len(group2id))

    batch_size = 16
    args = TrainingArguments(
        f"bert-finetuned-subgroup",
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=5,
        weight_decay=0.01,
        #load_best_model_at_end=True,
        #metric_for_best_model=metric_name,
    )


    # trainer
    trainer = Trainer(
        model,
        args,
        train_dataset=encoded_train_dataset,
        eval_dataset=encoded_validation_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    trainer.train()


    # testing
    predicts_unseen = trainer.predict(encoded_test_unseen_dataset)
    predicts_unseen_list = [np.argsort(pred)[::-1].tolist() for pred in predicts_unseen.predictions]
    predicts_seen = trainer.predict(encoded_test_seen_dataset)
    predicts_seen_list = [np.argsort(pred)[::-1].tolist() for pred in predicts_seen.predictions]

    # write to file
    write_test_groups_data(predicts_unseen_list, predicts_unseen_list, id2group, f'{args.output_dir}/bertchinese_unseen_subgroup.csv')
    print('unseen subgroup done')
    write_test_groups_data(predicts_seen_list, predicts_seen_list, id2group, f'{args.output_dir}/bertchinese_seen_subgroup.csv')
    print('seen subgroup done')
    


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--output_dir",
        type=Path,
        help="Path to the output directory.",
        default="./outputs/")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)





