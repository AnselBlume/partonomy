# PARTONOMY: Large Multimodal Models with Part-Level Visual Understanding

**PARTONOMY: Large Multimodal Models with Part-Level Visual Understanding**
\[*NeurIPS 2025 Spotlight*\🎖️]

*(\* co-first author)* 
[Ansel Blume*](https://anselblume.github.io/), [Jeonghwan Kim*](https://wjdghks950.github.io/), [Hyeonjeong Ha](https://hyeonjeongha.github.io/),  
[Elen Chatikyan](https://www.linkedin.com/in/elenchatikyan/), [Xiaomeng Jin](https://scholar.google.com/citations?user=Jd_tsuEAAAAJ&hl=en),  
[Khanh Duy Nguyen](https://scholar.google.com/citations?user=2RGZO6IAAAAJ&hl=en), [Nanyun Peng](https://violetpeng.github.io/),  
[Kai-Wei Chang](https://web.cs.ucla.edu/~kwchang/), [Derek Hoiem](https://dhoiem.cs.illinois.edu/), [Heng Ji](https://blender.cs.illinois.edu/hengji.html)

📄 **Paper:** [arXiv:2505.20759](https://arxiv.org/abs/2505.20759)

📄 **Dataset** [PARTONOMY-Core](https://huggingface.co/datasets/partonomy/partonomy-core)

---

## 🧭 Overview
In this work, we introduce **PARTONOMY**, a large-scale training dataset and benchmark for **pixel-level part grounding**, and **PLUM**, a segmentation-enabled Large Multimodal Model (LMM) designed for part-level visual understanding.

**PARTONOMY** provides 862 part labels and 534 object labels, offering rich hierarchical structure for evaluating models’ fine-grained reasoning capabilities. Unlike existing datasets that only ask models to recognize coarse, general parts (e.g., wheels), PARTONOMY focuses on *specialized concepts* (e.g., “agricultural airplane”) and challenge models to:

- Compare objects at the **part level**
- Reason about **part-whole relationships**
- Justify textual predictions in the pixel space with **visual segmentations**

To address architectural limitations in existing segmentation-enabled LMMs—such as reliance on `<SEG>` tokens unseen during pretraining and discarding previous segmentations—we propose **PLUM**, a novel LMM that:
- Uses **span tagging** instead of segmentation tokens, avoiding distribution shift. 
- Conditions on **previous predictions** in a feedback loop for iterative reasoning.

**PLUM** outperforms existing segmentation-enabled LMMs on reasoning segmentation, VQA, and visual hallucination benchmarks. When fine-tuned on our Explanatory Part Segmentation task, it performs competitively with models trained on much larger segmentation datasets.

---

## 🧩 Repository Structure
```
.
├── src
│   ├── models
│        ├── PLUM  # Our model code
              ├── scripts/
              │   ├── run_train_plum_0shot.sh       # Pretraining script
              │   ├── run_train_plum_ft.sh          # Fine-tuning script
              │   ├── run_validate_partonomy.sh     # Partonomy benchmark evaluation
              ├── utils/
              │   ├── explanatory_seg_dataset.py    # Dataset loading & preprocessing
              │   └── explanatory_dataset.py        # Data collation utilities
         ├── groundingLMM
         ├── LISA
         ├── PixelLM
         ├── segllm
```
`groundingLMM`, `LISA`, `PixelLM`, and `segllm` follow the same directory structure.

---

## ⚙️ Getting Started

### 1. Environment Setup
```bash
cd src/models/PLUM
conda env create -f environment.yml
conda activate partonomy
pip install flash-attn --no-build-isolation
```

### 2. Dataset Setup

To prepare the PARTONOMY dataset follow the steps down below:

## Training
### Training Data Preparation
The training data consists of 4 types of data:

1. Semantic segmentation datasets: [ADE20K](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip), [COCO-Stuff](http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip), [Mapillary](https://www.mapillary.com/dataset/vistas), [PACO-LVIS](https://github.com/facebookresearch/paco/tree/main#dataset-setup), [PASCAL-Part](https://github.com/facebookresearch/VLPart/tree/main/datasets#pascal-part), [COCO Images](http://images.cocodataset.org/zips/train2017.zip), [PartImageNet](https://huggingface.co/datasets/turkeyju/PartImageNet/blob/main/PartImageNet_Seg.zip)

    Note: For COCO-Stuff, we use the annotation file stuffthingmaps_trainval2017.zip. We only use the PACO-LVIS part in PACO. COCO Images should be put into the `dataset/coco/` directory.

3. Referring segmentation datasets: [refCOCO](https://web.archive.org/web/20220413011718/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip), [refCOCO+](https://web.archive.org/web/20220413011656/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip), [refCOCOg](https://web.archive.org/web/20220413012904/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip), [refCLEF](https://web.archive.org/web/20220413011817/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refclef.zip) ([saiapr_tc-12](https://web.archive.org/web/20220515000000/http://bvisionweb1.cs.unc.edu/licheng/referit/data/images/saiapr_tc-12.zip)) 

    Note: the original links of refCOCO series data are down, and we update them with new ones. If the download speed is super slow or unstable, we also provide a [OneDrive link](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155154502_link_cuhk_edu_hk/Em5yELVBvfREodKC94nOFLoBLro_LPxsOxNV44PHRWgLcA?e=zQPjsc) to download. **You must also follow the rules that the original datasets require.**

4. Visual Question Answering dataset: [LLaVA-Instruct-150k](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_instruct_150k.json)

5. Reasoning segmentation dataset: [ReasonSeg](https://github.com/dvlab-research/LISA#dataset)

Download them from the above links, and organize them as follows.

```
├── dataset  # put the `dataset` directory in the same directory as `weights`
│   ├── ade20k
│   │   ├── annotations
│   │   └── images
│   ├── coco
│   │   └── train2017
│   │       ├── 000000000009.jpg
│   │       └── ...
│   ├── cocostuff
│   │   └── train2017
│   │       ├── 000000000009.png
│   │       └── ...
│   ├── llava_dataset
│   │   └── llava_instruct_150k.json
│   ├── mapillary
│   │   ├── config_v2.0.json
│   │   ├── testing
│   │   ├── training
│   │   └── validation
│   ├── reason_seg
│   │   └── ReasonSeg
│   │       ├── train
│   │       ├── val
│   │       └── explanatory
│   ├── refer_seg
│   │   ├── images
│   │   |   ├── saiapr_tc-12 
│   │   |   └── mscoco
│   │   |       └── images
│   │   |           └── train2014
│   │   ├── refclef
│   │   ├── refcoco
│   │   ├── refcoco+
│   │   └── refcocog
│   ├── vlpart
│   │    ├── paco
│   │    │   └── annotations
│   │    └── pascal_part
│   │        ├── train.json
│   │        └── VOCdevkit
│   └── partimagenet
│   │    ├── descriptor.yaml
│   │    ├── graph.yaml
│   │    ├── PartImageNet
│   │    ├── partimagenet_qa_pairs.json
```

### Pre-trained weights

#### LLaVA-13B
To train PLUM-13B, you need to follow the [instruction](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md) to merge the LLaVA delta weights. For Llama2, we can directly use the LLaVA full weights `liuhaotian/llava-llama-2-13b-chat-lightning-preview`.

#### SAM ViT-H weights
Download SAM ViT-H pre-trained weights from the [link](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth). Put the SAM-ViT-H pre-trained weights in the `weights` directory (*the `weights` dir should be in the same directory as the above `dataset` directory*)

### 3. Pretraining PLUM

```bash
chmod +x scripts/run_train_plum_0shot.sh
./scripts/run_train_plum_0shot.sh
```

### 4. Fine-tuning on PARTONOMY

```bash
chmod +x scripts/run_train_plum_ft.sh
./scripts/run_train_plum_ft.sh
```

### 5. Evaluation – PARTONOMY

```bash
chmod +x scripts/run_validate_partonomy.sh
./scripts/run_validate_partonomy.sh
```

---

## 🧠 Citation

If you use this work, please cite:

```bibtex
@inproceedings{
  blume-kim-2025-partonomy,
  title={{PARTONOMY}: Large Multimodal Models with Part-Level Visual Understanding},
  author={Ansel Blume and Jeonghwan Kim and Hyeonjeong Ha and Elen Chatikyan and Xiaomeng Jin and Khanh Duy Nguyen and Nanyun Peng and Kai-Wei Chang and Derek Hoiem and Heng Ji},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
  url={https://openreview.net/forum?id=yjLew3Nd7z}
}
```

---

# License

Copyright 2025

Licensed under the **Apache License, Version 2.0 (the "License")**;
You may obtain a copy of the License at:

> [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an **"AS IS" BASIS,**
**WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,** either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

✅ **This repository and all included assets are fully open-sourced under the Apache License 2.0, permitting unrestricted commercial, academic, and industrial use.**