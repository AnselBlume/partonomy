'''
Script to count the number of lines in a file which do not start with "#" or "-"
'''
import re
import logging, coloredlogs

logger = logging.getLogger(__name__)

def count_annotated_objects(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    n_annotated = 0
    total = 0
    for line in lines:
        # Not a comment, not a listed part, but there is an inline comment
        if not re.match('\s+-', line) and not re.match('\s+#', line) and '#' in line:
            n_annotated += 1
            logger.debug(f'Line: {line}')

        total += 1

    return n_annotated, total

if __name__ == "__main__":
    coloredlogs.install(level='DEBUG')

    file_path = '/Users/Ansel/Documents/School/UIUC/research/sp24/ecole/dataset_paper/data/part_objaverse/dataset/part_notes.yaml'
    count, total = count_annotated_objects(file_path)
    print(f'Number of annotated objects: {count}/{total}')