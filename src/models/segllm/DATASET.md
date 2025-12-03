## Directory Structure
The dataset folders should be organized as follows. The following sections provides details on how to download the images, annotations and conversations data.
```
segllm
  ├── images_folder
  ├── annotations_folder
  ├── conversations_folder
```

## Images
Donwload images using the following links.

MSCOCO
```
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/train2014.zip
```

Reason Seg
```
gdown https://drive.google.com/drive/folders/125mewyg5Ao6tZ3ZdJ-1-E3n04LGVELqy --folder
```

Visual Genome
```
curl -o images.zip https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
curl -o images2.zip https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip
```

PASCAL
```
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar
```

ADE20K: See [instructions](https://ade20k.csail.mit.edu/request_data/).


The directory structue for `images_folder` should be organized as follows.
```
images_folder
  ├── ade20k
  │   └── images
  │       └── training
  ├── coco
  │   └── train2014
  │   └── val2014
  │   └── test2014
  ├── pascal
  ├── reason_seg
  │   └── images
  │       └── train
  │       └── val
  │       └── test
  ├── vg
  │   └── VG_100K
  │   └── VG_100K_2
```


## Conversations
Download the conversations data using git lfs. (To install git lfs, see instructions [here](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage?platform=mac))

Initialize git lfs and clone our custom-generated dataset repo from huggingface.
```
git lfs install
git clone https://huggingface.co/datasets/Marlo-Z/SegLLM_dataset
```

Extract data folders for our custom-generated conversations and annotations. `NOTE:` All conversations data are custom-generated. However, there are remaining off-the-shelf annotations need to be downloaded in the next step.
```
mv SegLLM_dataset/conversations_folder ./
mv SegLLM_dataset/annotations_folder ./
rm -rf SegLLM_dataset
```
The directory structure for `conversations_folder` is already setup from cloning the hugginface dataset repo.

## Annotations
Custom-generated annotations (`visual_genome` and `description_based_coco`) are already downloaded from the huggingface dataset repo via the previous step, and exist within the `annotations_folder`.

Off-the-shelf annotations can be downloaded from the following links - [refCOCO](https://web.archive.org/web/20220413011718/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip), [refCOCO+](https://web.archive.org/web/20220413011656/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip), [refCOCOg](https://web.archive.org/web/20220413012904/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip), [ReasonSeg](https://github.com/dvlab-research/LISA#dataset), [ADE20K](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip), [LVIS](https://www.lvisdataset.org/dataset), [PASCAL-Part](https://github.com/pmeletis/panoptic_parts), and [PACO-LVIS](https://github.com/facebookresearch/paco/tree/main#dataset-setup)

The directory structure for the `annotations_folder` should be organized as follows.
```
annotations_folder
  ├── ade20k
  │   └── annotations
  │       └── training
  ├── cocostuff
  │   └── train2017
  │   └── val2017
  ├── description_based_coco
  │   └── seg_mask_per_instance.json
  ├── lvis
  │   └── lvis_v1_train.json
  ├── paco_lvis
  │   └── annotations
  │       └── paco_lvis_v1_train.json
  │       └── paco_lvis_v1_val.json
  ├── pascal
  │   └── pascal_labels.json
  ├── reason_seg
  │   └── annotations
  │       └── train
  │       └── val
  │       └── test
  ├── refcoco
  │   └── instances.json
  ├── refcoco+
  │   └── instances.json
  ├── refcocog
  │   └── instances.json
  ├── visual_genome
  │   └── vg_masks_train_new.json
```