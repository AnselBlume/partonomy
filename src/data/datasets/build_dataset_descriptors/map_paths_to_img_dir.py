import os
import logging
import orjson
from tqdm import tqdm
from typing import List, Union

logger = logging.getLogger(__name__)

class PathMapper:
    def __init__(self, old_img_dirs: Union[List[str], str], new_img_dirs: Union[List[str], str], warn_missing_paths: bool = True):
        # Convert single directories to lists for uniform handling
        self.old_img_dirs = [old_img_dirs] if isinstance(old_img_dirs, str) else old_img_dirs
        self.new_img_dirs = [new_img_dirs] if isinstance(new_img_dirs, str) else new_img_dirs

        # Ensure the lists have the same length
        if len(self.old_img_dirs) != len(self.new_img_dirs):
            raise ValueError("The number of old and new image directories must match")

        self.warn_missing_paths = warn_missing_paths

    def map_path(self, path: str):
        for i, old_img_dir in enumerate(self.old_img_dirs):
            try:
                # Check if we can create a valid relative path
                rel_path = os.path.relpath(path, old_img_dir)
                # If the relative path starts with '..' it means the path is not within the directory
                if not rel_path.startswith('..'):
                    new_path = os.path.join(self.new_img_dirs[i], rel_path)

                    # Check if the new path exists
                    if self.warn_missing_paths and not os.path.exists(new_path):
                        logger.debug(f'New path {new_path} does not exist; trying to fix')

                        # NOTE: Hack to map PACO_LVIS images; try appending -1
                        new_path = new_path.replace('.jpg', '-1.jpg')

                        if not os.path.exists(new_path):
                            logger.warning(f'New path {new_path} does not exist')
                            continue
                        # End remove block

                        logger.warning(f'New path {new_path} does not exist')

                    return new_path
            except ValueError:
                # This happens when the path and directory are on different drives
                continue

        # If we get here, no valid relative path was found
        logger.warning(f'Could not find a valid relative path for {path} in any of the provided directories')
        return path

    def _find_path_in_dir(self, path: str, dir: str):
        # TODO more complex checking, handling the case where the original path in the directory points to
        # the path because it's a hard/symlink
        # For now, just return the path
        return path

    def map_qa_pairs(self, qa_pairs: list[dict]):
        for qa_pair in tqdm(qa_pairs, desc='Mapping QA pairs'):
            qa_pair['image_path'] = self.map_path(qa_pair['image_path'])

        return qa_pairs

if __name__ == '__main__':
    from jsonargparse import ArgumentParser
    import coloredlogs

    coloredlogs.install(level='DEBUG')

    parser = ArgumentParser()
    parser.add_argument('--old_img_dirs', type=str, nargs='+', required=True, help='List of old image directories to search for relative paths')
    parser.add_argument('--new_img_dirs', type=str, nargs='+', required=True, help='List of new image directories corresponding to old_img_dirs')
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--indent', type=bool, default=False)
    parser.add_argument('--warn_missing_paths', type=bool, default=True)

    args = parser.parse_args()

    if len(args.old_img_dirs) != len(args.new_img_dirs):
        logger.error("The number of old and new image directories must match")
        exit(1)

    if args.output_file is None:
        args.output_file = os.path.splitext(args.input_file)[0] + '_mapped.json'
        logger.info(f'Output file not provided, using {args.output_file}')

    logger.info(f'Loading QA pairs from {args.input_file}')
    with open(args.input_file, 'rb') as f:
        qa_pairs = orjson.loads(f.read())

    mapper = PathMapper(args.old_img_dirs, args.new_img_dirs, args.warn_missing_paths)
    mapped_qa_pairs = mapper.map_qa_pairs(qa_pairs)

    with open(args.output_file, 'wb') as f:
        f.write(orjson.dumps(mapped_qa_pairs, option=orjson.OPT_INDENT_2 if args.indent else None))

    logger.info(f'Saved mapped QA pairs to {args.output_file}')