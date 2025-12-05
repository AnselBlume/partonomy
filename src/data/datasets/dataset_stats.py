# %%
import orjson
import spacy
from tqdm import tqdm
from typing import List, Dict

nlp = spacy.load("en_core_web_sm")
SEP = '--part:'

def join_object_and_part_names(object_name: str, part_name: str) -> str:
    return f'{object_name}{SEP}{part_name}'

def batch_lemmatize(part_names: List[str]) -> Dict[str, str]:
    """Batch lemmatize a list of part names and return a map from original to lemmatized form."""
    lemmatized_map = {}
    for doc, original in zip(nlp.pipe(part_names, batch_size=1000), part_names):
        lemmatized_map[original] = " ".join([token.lemma_ for token in doc])
    return lemmatized_map

def _get_qa_pairs_data(qa_pairs: List[Dict]) -> Dict:
    image_paths = set()
    object_labels = set()
    part_labels = set()
    object_part_labels = set()
    n_segmentations = 0

    path_part_pairs_encountered = set()

    # Collect all part names to lemmatize only once
    print(f'Lemmatizing {len(qa_pairs)} qa pairs')
    all_part_names = {part_name for qa in qa_pairs for part_name in qa['segmentations']}
    part_lemma_map = batch_lemmatize(list(all_part_names))

    for qa_pair in tqdm(qa_pairs):
        image_paths.add(qa_pair['image_path'])
        object_label = qa_pair['image_label']
        object_labels.add(object_label)

        for part_name, part_segs in qa_pair['segmentations'].items():
            part_label = part_lemma_map[part_name]
            part_labels.add(part_label)

            object_part_label = join_object_and_part_names(object_label, part_label)
            object_part_labels.add(object_part_label)

            # There are multiple question types in a file, and each of these questions can have different
            # part segmentations. So try to normalize them by counting each path-part label pair once
            path_part_pair = (qa_pair['image_path'], object_part_label)
            if path_part_pair not in path_part_pairs_encountered:
                path_part_pairs_encountered.add(path_part_pair)
                n_segmentations += len(part_segs)

    return {
        'part_labels': part_labels,
        'object_labels': object_labels,
        'object_part_labels': object_part_labels,
        'image_paths': image_paths,
        'n_segmentations': n_segmentations
    }

def get_qa_pairs_data(qa_pairs_paths: List[str]) -> Dict:
    image_paths = set()
    object_labels = set()
    part_labels = set()
    object_part_labels = set()
    n_segmentations = 0

    print(f'Processing {len(qa_pairs_paths)} qa pairs files')
    for qa_pairs_path in qa_pairs_paths:
        print(f'Processing {qa_pairs_path}')
        with open(qa_pairs_path, 'rb') as f:
            qa_pairs = orjson.loads(f.read())

        data = _get_qa_pairs_data(qa_pairs)

        image_paths.update(data['image_paths'])
        object_labels.update(data['object_labels'])
        part_labels.update(data['part_labels'])
        object_part_labels.update(data['object_part_labels'])
        n_segmentations += data['n_segmentations']

    return {
        'image_paths': image_paths,
        'object_labels': object_labels,
        'part_labels': part_labels,
        'object_part_labels': object_part_labels,
        'n_segmentations': n_segmentations
    }

def print_stats(data: dict):
    print(f'Number of images: {len(data["image_paths"])}')
    print(f'Number of objects: {len(data["object_labels"])}')
    print(f'Number of parts: {len(data["part_labels"])}')
    print(f'Number of object-part pairs: {len(data["object_part_labels"])}')
    print(f'Number of segmentations: {data["n_segmentations"]}')

# %%
if __name__ == '__main__':
    train_paths = []
    val_paths = []

    for dataset_name in ['paco_lvis', 'partimagenet', 'partonomy', 'pascal_part']:
        if dataset_name != 'partonomy':
            qa_pairs_train_path = f'/shared/nas2/blume5/sp25/partonomy/partonomy_private/data/partonomy_descriptors/{dataset_name}/{dataset_name}_qa_pairs_train.json'
            train_paths.append(qa_pairs_train_path)

        qa_pairs_val_path = f'/shared/nas2/blume5/sp25/partonomy/partonomy_private/data/partonomy_descriptors/{dataset_name}/{dataset_name}_qa_pairs_val.json'
        val_paths.append(qa_pairs_val_path)

    train_data = get_qa_pairs_data(train_paths)
    val_data = get_qa_pairs_data(val_paths)

    print("Train stats:")
    print_stats(train_data)
    print("\nValidation stats:")
    print_stats(val_data)

    print('\nComputing global stats')
    global_data = {
        'image_paths': train_data['image_paths'] | val_data['image_paths'],
        'object_labels': train_data['object_labels'] | val_data['object_labels'],
        'part_labels': train_data['part_labels'] | val_data['part_labels'],
        'object_part_labels': train_data['object_part_labels'] | val_data['object_part_labels'],
        'n_segmentations': train_data['n_segmentations'] + val_data['n_segmentations']
    }

    print('Global stats:')
    print_stats(global_data)

    # Separately compute the number of segmentations from the PartDatasetDescriptors

# %%
