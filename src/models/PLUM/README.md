# PARTONOMY: Large Multimodal Models with Part-Level Visual Understanding

---

**PARTONOMY: Large Multimodal Models with Part-Level Visual Understanding [[Paper](https://arxiv.org/abs/2505.20759)]** <br />
(\* co-first author)
[Ansel Blume*](https://anselblume.github.io/),
[Jeonghwan Kim*](https://wjdghks950.github.io/),
[Hyeonjeong Ha](https://hyeonjeongha.github.io/),
[Elen Chatikyan](https://www.linkedin.com/in/elenchatikyan/),
[Xiaomeng Jin](https://scholar.google.com/citations?user=Jd_tsuEAAAAJ&hl=en),
[Khanh Duy Nguyen](https://scholar.google.com/citations?user=2RGZO6IAAAAJ&hl=en),
[Nanyun Peng](https://violetpeng.github.io/),
[Kai-Wei Chang](https://web.cs.ucla.edu/~kwchang/),
[Derek Hoiem](https://dhoiem.cs.illinois.edu/),
[Heng Ji](https://blender.cs.illinois.edu/hengji.html),<br />

## Overview
In this work, we introduce **PARTONOMY**, an LMM benchmark designed for pixel-level part grounding, and **PLUM**, a part-understanding, segmentation-enabled Large Multimodal Model (LMM). PARTONOMY encompasses a totla of 862 part labels and 534 object labels for evaluation. Unlike existing datasets that simply ask models to identify generic parts, PARTONOMY uses specialized concepts (e.g.,  agricultural airplane), and challenges models to compare objects' parts, consider part-whole relationships, and justify textual predictions with visual segmentations.

We also note that existing segmentation-enabled LMMs have two key architectural shortcomings: they use special \seg tokens not seen during pretraining which induce distribution shift, and they discard predicted segmentations instead of using past predictions to guide future ones. To address these deficiencies, we propose **PLUM**, a novel segmenting LMM that uses span tagging instead of segmentation tokens and that conditions on prior predictions in a feedback loop. We find that pretrained PLUM outperforms existing segmenting LMMs on reasoning segmentation, VQA, and visual hallucination benchmarks. In addition, PLUM finetuned on our proposed Explanatory Part Segmentation task is competitive with segmenting LMMs trained on significantly more segmentation data. Our work opens up new avenues towards enabling fine-grained, grounded visual understanding in LMMs.

---

## Repository Structure

- **`validate_partonomy.py`**Contains the evaluation logic. This script loads the evaluation dataset, runs the segmentation model, computes metrics (e.g., gIoU, cIoU), and outputs results.*For evaluation, run the provided script `run_validate_partonomy.sh`.*
- **`run_validate_partonomy.sh`**A shell script to execute the validation pipeline easily. It wraps around `validate_partonomy.py` with the appropriate arguments.
- **`utils/explanatory_seg_dataset.py`**Implements the `ExplanatorySegDataset` class, which is responsible for loading and preprocessing the Partonomy evaluation dataset. It includes functionality for handling image preprocessing, segmentation mask loading, and preparing conversation prompts.
- **`utils/explanatory_dataset.py`**Contains the `collate_fn` function used to batch data samples from `ExplanatorySegDataset` before passing them to the model for evaluation.
- **Other Files/Directories:**
  Additional modules, configuration files, and scripts necessary for training, model definition, and utility functions.

---

## Getting Started

### Requirements

#### Environment Setup
Install the required packages and dependencies with the following command:
```bash
conda create --name partonomy --file requirements.txt
```

#### Dataset Setup: Image Download
TODO

### Pre-Training
```bash
chmod +x scripts/run_train_plum_0shot.sh
./run_train_plum_0shot.sh
```

### Finetuning on Partonomy
```bash
chmod +x scripts/run_train_plum_ft.sh
./run_train_plum_ft.sh
```

### Evaluation - Partonomy
```bash
chmod +x scripts/run_validate_partonomy.sh
./run_validate_partonomy.sh
```

### Evaluation - Referring Expression Segmentation
```bash
chmod +x scripts/run_validate_seg.sh
./run_validate_seg.sh
```

