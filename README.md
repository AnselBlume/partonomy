# PARTONOMY: Large Multimodal Models with Part-Level Visual Understanding

**PARTONOMY: Large Multimodal Models with Part-Level Visual Understanding**  
*(\* co-first author)*  
[Ansel Blume*](https://anselblume.github.io/), [Jeonghwan Kim*](https://wjdghks950.github.io/), [Hyeonjeong Ha](https://hyeonjeongha.github.io/),  
[Elen Chatikyan](https://www.linkedin.com/in/elenchatikyan/), [Xiaomeng Jin](https://scholar.google.com/citations?user=Jd_tsuEAAAAJ&hl=en),  
[Khanh Duy Nguyen](https://scholar.google.com/citations?user=2RGZO6IAAAAJ&hl=en), [Nanyun Peng](https://violetpeng.github.io/),  
[Kai-Wei Chang](https://web.cs.ucla.edu/~kwchang/), [Derek Hoiem](https://dhoiem.cs.illinois.edu/), [Heng Ji](https://blender.cs.illinois.edu/hengji.html)

📄 **Paper:** [arXiv:2505.20759](https://arxiv.org/abs/2505.20759)

* Code and dataset will be released soon!

---

## 🧭 Overview
In this work, we introduce **PARTONOMY**, a large-scale benchmark for **pixel-level part grounding**, and **PLUM**, a segmentation-enabled Large Multimodal Model (LMM) designed for part-level visual understanding.

**PARTONOMY** provides 862 part labels and 534 object labels, offering rich hierarchical structure for evaluating models’ fine-grained reasoning capabilities. Unlike existing datasets that only ask models to recognize coarse parts, PARTONOMY focuses on *specialized concepts* (e.g., “agricultural airplane”) and challenges models to:

- Compare objects at the **part level**
- Reason about **part-whole relationships**
- Justify textual predictions with **visual segmentations**

To address architectural limitations in existing segmentation-enabled LMMs—such as reliance on `<SEG>` tokens unseen during pretraining and discarding previous segmentations—we propose **PLUM**, a novel LMM that:
- Uses **span tagging** instead of segmentation tokens, avoiding distribution shift.  
- Conditions on **previous predictions** in a feedback loop for iterative reasoning.

**PLUM** outperforms existing segmentation-enabled LMMs on reasoning segmentation, VQA, and visual hallucination benchmarks. When fine-tuned on our Explanatory Part Segmentation task, it performs competitively with models trained on much larger segmentation datasets.

---

## 🧩 Repository Structure
```
.
├── scripts/
│   ├── run_train_plum_0shot.sh       # Pretraining script
│   ├── run_train_plum_ft.sh          # Fine-tuning script
│   ├── run_validate_partonomy.sh     # Partonomy benchmark evaluation
│   └── run_validate_seg.sh           # Referring expression segmentation eval
│
├── utils/
│   ├── explanatory_seg_dataset.py    # Dataset loading & preprocessing
│   └── explanatory_dataset.py        # Data collation utilities
│
├── validate_partonomy.py             # Evaluation logic
├── requirements.txt                  # Dependencies
└── LICENSE                           # License text (Apache 2.0)
```

---

## ⚙️ Getting Started

### 1. Environment Setup
```bash
conda create --name partonomy --file requirements.txt
conda activate partonomy
````

### 2. Dataset Setup

To prepare the PARTONOMY dataset:

1. Download images and segmentation masks from the provided dataset links (coming soon).

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

### 6. Evaluation – Referring Expression Segmentation

```bash
chmod +x scripts/run_validate_seg.sh
./scripts/run_validate_seg.sh
```

---

## 🧠 Citation

If you use this work, please cite:

```bibtex
@article{blume-kim-2025partonomy,
  title={PARTONOMY: Large Multimodal Models with Part-Level Visual Understanding},
  author={Blume*, Ansel and Kim*, Jeonghwan and Ha, Hyeonjeong and Chatikyan, Elen and Jin, Xiaomeng and Nguyen, Khanh Duy and Peng, Nanyun and Chang, Kai-Wei and Hoiem, Derek and Ji, Heng},
  journal={https://arxiv.org/abs/2505.20759v3},
  year={2025}
}
```

---

# 🪪 License

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