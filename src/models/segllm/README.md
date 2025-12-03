# SegLLM: Multi-round Reasoning Segmentation

We present SegLLM, a novel multi-round interactive segmentation model that leverages conversational memory of both visual and textual outputs to reason over previously segmented objects and past interactions, effectively interpreting complex user intentions.

<p align="center"> 
  <img width="1301" alt="demo" src=./assets/demo.gif align="center" >
</p>

> [**SegLLM: Multi-round Reasoning Segmentation**](http://arxiv.org/abs/2410.18923)            
> [XuDong Wang*](https://frank-xwang.github.io/), Shaolun Zhang*, [Shufan Li*](https://homepage.jackli.org/), [Konstantinos Kallidromitis](https://tech-ai.panasonic.com/en/researcher_introduction/048/), Kehan Li, Yusuke Kato, Kazuki Kozuka, [Trevor Darrell](https://people.eecs.berkeley.edu/~trevor/)       
> UC Berkeley, UCLA, Panasonic AI Research, Stanford            
> ICLR 2025         

[[`project page`](https://berkeley-hipie.github.io/segllm.github.io/)] [[`arxiv`](https://arxiv.org/pdf/2410.18923)] [[`bibtex`](#citation)] [[`Huggingface`](https://huggingface.co/Marlo-Z/SegLLM/tree/main)]          


## Updates
- 01/22/2025 SegLLM was accepted by ICLR 2025!!!
- 12/29/2024 Release model training codes and datasets.
- 11/05/2024 Release model evaluation codes.
- 11/03/2024 Initial commit: release model inference codes and Gradio demo.

## Installation and Dataset
See [installation instructions](./INSTALL.md) and [dataset setup instructions](./DATASET.md).

## Inference

<p align="center"> 
  <img width="1301" alt="pipeline" src=./assets/architecture.png align="center" >
</p>

Launch the Gradio demo:
```
CUDA_VISIBLE_DEVICES=0 ./scripts/inference/launch_gradio_demo.sh
```
Launch inference via command line:
```
CUDA_VISIBLE_DEVICES=0 ./scripts/inference/launch_cli_demo.sh
```
Consider trying the example images and conversations in `inference_images`.

## Evaluation

<p align="center"> 
  <img width="1301" alt="demo" src=./assets/mr_refcoco_table.png align="center" >
</p>

To evaluate on the following datasets, respectively: multi-round RefCOCO, single-round RefCOCO, single-round RefCOCO with different question templates, multi-round PACO and ReasonSeg:
```
LOCAL_HOST=0 ./scripts/eval/eval_mr_refcoco.sh
LOCAL_HOST=0 ./scripts/eval/eval_refcoco.sh
LOCAL_HOST=0 ./scripts/eval/eval_refcoco_templates.sh
LOCAL_HOST=0 ./scripts/eval/eval_mr_paco.sh
LOCAL_HOST=0 ./scripts/eval/eval_reason_seg.sh
```

## Training
To reproduce our MR-RefCOCO checkpoint, MR-PACO checkpoint, and all-datasets checkpoint, respectively, run the following commands:
```
LOCAL_HOST=0,1,2,3 ./scripts/train/train_mr_refcoco.sh
LOCAL_HOST=0,1,2,3 ./scripts/train/train_mr_paco.sh
LOCAL_HOST=0,1,2,3 ./scripts/train/train_all_data_mix.sh
```

## Checkpoints
The model checkpoints are available at [Huggingface](https://huggingface.co/Marlo-Z/SegLLM/tree/main)



## Citation
If you find our work inspiring or use our codebase in your research, please consider giving a star ⭐ and a citation.
```
@article{wang2024segllm,
  title={SegLLM: Multi-round Reasoning Segmentation},
  author={Wang, XuDong and Zhang, Shaolun and Li, Shufan and Kallidromitis, Konstantinos and Li, Kehan and Kato, Yusuke and Kozuka, Kazuki and Darrell, Trevor},
  journal={arXiv preprint arXiv:2410.18923},
  year={2024}
}
```
