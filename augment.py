import argparse
import json
from typing import List

from datasets import load_dataset
from textattack.augmentation import EmbeddingAugmenter

SPLIT = "train"


def convert_rte(dataset: List[dict]):
    id_list = []
    sent1_list = []
    sent2_list = []
    label_list = []
    for example_dict in dataset:
        id_list.append(example_dict["idx"])
        sent1_list.append(example_dict["sentence1"])
        sent2_list.append(example_dict["sentence2"])
        label_list.append(example_dict["label"])
    return id_list, sent1_list, sent2_list, label_list


def augment_examples(sentences: List[str], num_transformations: int = 1):
    emb_augmenter = EmbeddingAugmenter(transformations_per_example=num_transformations)
    augmented = emb_augmenter.augment_many(sentences, show_progress=True)
    augmented = [element[0] for element in augmented]
    return augmented


def augment_rte(only_short: bool = False) -> List[dict]:
    dataset = load_dataset(path="glue", name="rte", split=SPLIT)
    dataset = list(dataset)
    id_list, sent1_list, sent2_list, label_list = convert_rte(dataset)

    last_idx = max(id_list) + 1

    if only_short:
        sent1_aug = sent1_list
    else:
        # this will take a while
        sent1_aug = augment_examples(sent1_list)
    sent2_aug = augment_examples(sent2_list)
    augmented_dataset = []
    for idx, sentence1, sentence2, label in zip(id_list, sent1_aug, sent2_aug, label_list):
        augmented_dataset.append({"idx": idx + last_idx,
                                  "sentence1": sentence1,
                                  "sentence2": sentence2,
                                  "label": label})
    dataset.extend(augmented_dataset)
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment specified HuggingFace dataset")
    parser.add_argument("-p", "--path", help="Dataset path", type=str, default="glue")
    parser.add_argument("-n", "--name", help="Dataset name", type=str, default="rte")
    parser.add_argument("--only_short", help="Augment only short examples", type=bool, default=True)
    parser.add_argument("--write_path", help="Path to write the result", type=str, default="augmented.jsonl")
    args = parser.parse_args()

    if args.path == "glue" and args.name == "rte":
        augmented = augment_rte(only_short=args.only_short)

    else:
        raise NotImplementedError

    with open(args.write_path, "w") as file:
        for example in augmented:
            file.write(json.dumps(example) + "\n")
