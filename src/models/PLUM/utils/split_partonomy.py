import os
import yaml
import random
from collections import defaultdict

DATASET_DIR = "../dataset"

def main():
    random.seed(42)

    with open(os.path.join(DATASET_DIR, "partonomy", "graph.yaml"), "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    concepts = data["concepts"]
    superordinate_to_coarse = defaultdict(lambda: defaultdict(list))

    for concept in concepts:
        parts = concept.split("--")  # split e.g. "airplanes--agricultural--part:cockpit" -> ["airplanes", "agricultural", "part:cockpit"]
        superordinate = parts[0]
        if len(parts) == 1:
            coarse = superordinate
        else:
            coarse = "--".join(parts[0:2])
        superordinate_to_coarse[superordinate][coarse].append(concept)

    train_val_ratio = 0.8
    train_dict = defaultdict(list)
    val_dict = defaultdict(list)

    for superordinate, coarse_dict in superordinate_to_coarse.items():
        all_coarse_concepts = list(coarse_dict.keys())
        random.shuffle(all_coarse_concepts)

        n_coarse = len(all_coarse_concepts)
        split_index = int(train_val_ratio * n_coarse)

        coarse_train = all_coarse_concepts[:split_index]
        coarse_val = all_coarse_concepts[split_index:]

        for c_train in coarse_train:
            train_dict[superordinate].extend(coarse_dict[c_train])
        for c_val in coarse_val:
            val_dict[superordinate].extend(coarse_dict[c_val])

    print("Superordinate | Total | Train | Validation")

    for superordinate in sorted(superordinate_to_coarse.keys()):
        total_concepts = len(train_dict[superordinate]) + len(val_dict[superordinate])
        n_train = len(train_dict[superordinate])
        n_val = len(val_dict[superordinate])
        print(f"{superordinate} | {total_concepts} | {n_train} | {n_val}")


    with open(os.path.join(DATASET_DIR, "partonomy", "train_concepts.txt"), "w", encoding="utf-8") as ftrain, \
         open(os.path.join(DATASET_DIR, "partonomy", "val_concepts.txt"), "w", encoding="utf-8") as fval:
        for sup in sorted(train_dict.keys()):
            for c in train_dict[sup]:
                ftrain.write(c + "\n")
        for sup in sorted(val_dict.keys()):
            for c in val_dict[sup]:
                fval.write(c + "\n")

if __name__ == "__main__":
    main()
