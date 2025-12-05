#! /bin/bash

# Pascal Part
# python /work/hdd/beig/blume5/partonomy/partonomy_private/src/data/datasets/build_dataset_descriptors/map_paths_to_img_dir.py \
#     --old_img_dir '/shared/nas/data/m1/jk100/code/OpenAttrLibrary/LISA/dataset/vlpart/pascal_part/VOCdevkit/VOC2010/JPEGImages' \
#     --new_img_dir '/work/hdd/beig/blume5/partonomy/data/vlpart/pascal_part/VOCdevkit/VOC2010/JPEGImages' \
#     --input_file '/work/hdd/beig/blume5/partonomy/data/partonomy_descriptors/pascal_part/pascal_part_qa_pairs_train.json' \
#     --output_file '/work/hdd/beig/blume5/partonomy/data/partonomy_descriptors/pascal_part/delta-pascal_part_qa_pairs_train.json' \
#     --indent false \
#     --warn_missing_paths true

# PACO
python /work/hdd/beig/blume5/partonomy/partonomy_private/src/data/datasets/build_dataset_descriptors/map_paths_to_img_dir.py \
    --old_img_dir '/shared/nas/data/m1/jk100/code/OpenAttrLibrary/LISA/dataset/coco/train2017' '/shared/nas2/blume5/sp25/partonomy/partonomy_private/data/modified_images' \
    --new_img_dir '/work/hdd/beig/blume5/partonomy/data/coco2017_jpg/jk100/train2017' '/work/hdd/beig/blume5/partonomy/data/coco2017_jpg/modified_images' \
    --input_file '/work/hdd/beig/blume5/partonomy/data/partonomy_descriptors/paco_lvis/paco_lvis_qa_pairs_val.json' \
    --output_file '/work/hdd/beig/blume5/partonomy/data/partonomy_descriptors/paco_lvis/delta-paco_lvis_qa_pairs_val.json' \
    --indent false \
    --warn_missing_paths true

    # --new_img_dir '/work/hdd/beig/blume5/partonomy/data/glamm/coco_2017/train2017' '/work/hdd/beig/blume5/partonomy/data/modified_images' \

# PartImageNet
# python /work/hdd/beig/blume5/partonomy/partonomy_private/src/data/datasets/build_dataset_descriptors/map_paths_to_img_dir.py \
#     --old_img_dir '/shared/nas2/blume5/sp25/partonomy/partonomy_private/data/dataset/partimagenet/PartImageNet/images/train' \
#     --new_img_dir '/work/hdd/beig/blume5/partonomy/data/PartImageNet/images/train' \
#     --input_file '/work/hdd/beig/blume5/partonomy/data/partonomy_descriptors/partimagenet/partimagenet_qa_pairs_val.json' \
#     --output_file '/work/hdd/beig/blume5/partonomy/data/partonomy_descriptors/partimagenet/delta-partimagenet_qa_pairs_val.json' \
#     --indent false \
#     --warn_missing_paths true