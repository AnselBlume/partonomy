# %%
from jsonargparse import Namespace, ArgumentParser
import sys
sys.path.append('/shared/nas2/blume5/sp25/partonomy/partonomy_private/src') # Runnability in ipynb
import os
import torch
import pycocotools.mask as mask_utils
import json
import numpy as np
from PIL.Image import Image as PILImage
from tqdm import tqdm
from rembg import remove, new_session
from sam2.sam2_image_predictor import SAM2ImagePredictor
from data.partonomy_core.annotation_completion.heatmap import sample_points, extract_components
from dataclasses import dataclass
# from .heatmap import sample_points, extract_components
from typing import Union
from data.utils import open_image, list_paths
from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image, pil_to_tensor
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

mask_color = np.array([30/255, 144/255, 255/255, 0.6]) # Blue; RGBA

# Visualization methods from from https://github.com/facebookresearch/sam2/blob/main/notebooks/image_predictor_example.ipynb
def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = mask_color
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='red', facecolor=(0, 0, 0, 0), lw=3))

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    figs = []
    for i, (mask, score) in enumerate(zip(masks, scores)):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        # if len(scores) > 1:
        ax.set_title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        ax.axis('off')

        figs.append((fig, ax))

    return figs

def bbox_from_mask(masks: Union[torch.Tensor, np.ndarray], use_dim_order: bool = False) -> Union[torch.IntTensor, np.ndarray]:
    '''
        Borrowed from: https://github.com/AnselBlume/ecole_mo9_demo/blob/main/src/image_processing/localize.py
        Given a set of masks, return the bounding box coordinates and plot the bounding box with width bbox_width.

        ### Arguments:
            masks: (n,h,w)
            use_dim_order (Optional[bool]): Use the order of the dimensions of the masks when returning points (i.e.
                height dim, width dim, height dim, width dim). This is as opposed to XYXY, which corresponds to
                (width dim, height dim, width dim, height dim).

        ### Returns:
            (n,4) bounding box tensor or ndarray, depending on input type. First two coordinates specify upper left
            corner, while next two coordinates specify bottom right corner. Return format determined by value of use_dim_order.

        ### Note
        XYXY order can be visualized with:

            ```python
            import matplotlib.pyplot as plt
            from torchvision.utils import draw_bounding_boxes

            img = torch.zeros(3, 300, 500).int()
            boxes = torch.tensor([[50, 100, 300, 275]])

            plt.imshow(draw_bounding_boxes(img, boxes=boxes).permute(1, 2, 0))
            ```

        NOTE: This is also implemented in torchvision.ops.masks_to_boxes, but their implementation is not vectorized wrt
        number of masks.
    '''
    is_np = isinstance(masks, np.ndarray)
    if is_np:
        masks = torch.from_numpy(masks)

    if len(masks) == 0:
        return torch.zeros(0, 4).int() if not is_np else np.zeros((0, 4), dtype=int)

    # Convert masks to boolean and zero pad the masks to have regions on image edges
    # use the edges as boundaries
    masks = masks.bool()

    # Find pixels bounding mask
    top_inds = (
        masks.any(dim=2) # Which rows have nonzero values; (n,h)
             .int() # Argmax not implemented for bool, so convert back to int
             .argmax(dim=1) # Get the first row with nonzero values; (n,)
    )

    left_inds = masks.any(dim=1).int().argmax(dim=1) # Get the first column with nonzero values; (n,)
    bottom_inds = masks.shape[1] - 1 - masks.flip(dims=(1,)).any(dim=2).int().argmax(dim=1) # Reverse rows to get last row with nonzero vals; (n,)
    right_inds = masks.shape[2] - 1 - ( # Since reversing, subtract from total width
        masks.flip(dims=(2,)).any(dim=1) # Reverse columns to get last column with nonzero vals; (n,h)
             .int() # Argmax not implemented for bool, so convert back to int
             .argmax(dim=1) # Get the first column with nonzero values; (n,)
    )

    if use_dim_order: # Specify UL, BR by order of image dimensions: (h, w, h, w)
        upper_lefts = torch.cat([top_inds[:, None], left_inds[:, None]], dim=1) # (n,2)
        bottom_rights = torch.cat([bottom_inds[:, None], right_inds[:, None]], dim=1) # (n,2)

    else: # Specify UL, BR by order of XYXY: (w, h, w, h)
        upper_lefts = torch.cat([left_inds[:, None], top_inds[:, None]], dim=1) # (n,2)
        bottom_rights = torch.cat([right_inds[:, None], bottom_inds[:, None]], dim=1) # (n,2)

    boxes = torch.cat([upper_lefts, bottom_rights], dim=1).int() # (n,4)

    if is_np:
        boxes = boxes.numpy()

    return boxes

def load_mask(rle_path: str):
    with open(rle_path, 'r') as f:
        rle_dict = json.load(f)

    mask = mask_utils.decode(rle_dict).astype(np.uint8)
    image_path = rle_dict['image_path']

    return mask, image_path

def test_region_prediction():
    '''
    Initial testing code with a couple of good examples of region prediction
    '''
    predictor = SAM2ImagePredictor.from_pretrained('facebook/sam2-hiera-large')

    # rle_path = '/shared/nas2/blume5/sp25/partonomy/partonomy_private/data/partonomy_part_heatmaps/airplanes--transport--part:propulsion component/0a472b4c258f9c24f597cad3a205e21f8268c6e53fe50cc70df2b1f56582de83.json'
    rle_path = '/shared/nas2/blume5/sp25/partonomy/partonomy_private/data/partonomy_part_heatmaps/garden--bulb planter--part:foot pedal/9aa3e35436995590070591077a664c9f.json'
    mask, image_path = load_mask(rle_path)
    components = extract_components(mask) # (n, h, w)

    for i, component in enumerate(components):
        pos_points, neg_points = sample_points(
            component,
            num_positive=10,
            num_negative=10,
            min_dist_from_mask=100,
            bias_positive=True,
            bias_negative=False
        )

        pos_points = pos_points[:, ::-1] # (n, 2) from row, col (y,x) --> (x,y)
        neg_points = neg_points[:, ::-1] # (n, 2) from row, col (y,x) --> (x,y)

        point_coords = np.concatenate([pos_points, neg_points], axis=0)
        point_labels = np.array([1] * pos_points.shape[0] + [0] * neg_points.shape[0])

        box_xyxy = bbox_from_mask(component[None,...]).squeeze() # (4,) in xyxy

        image = open_image(image_path)

        # Draw component mask
        image_t = pil_to_tensor(image)
        mask_t = torch.from_numpy(component)[None,...].bool()

        superimposed_image_t = draw_segmentation_masks(image_t, mask_t, alpha=0.7, colors=tuple(mask_color*255)[:-1])
        superimposed_image_t = draw_bounding_boxes(superimposed_image_t, torch.from_numpy(box_xyxy[None,...]), colors='red', width=2)
        superimposed_image = to_pil_image(superimposed_image_t)

        fig, ax = plt.subplots(figsize=(10, 15))
        ax.imshow(superimposed_image)
        ax.axis('off')
        ax.set_title(f'Component mask: {i+1}')
        plt.show()
        plt.close(fig)

        # Visualize SAM result
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            predictor.set_image(image)
            masks, scores, logits = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box_xyxy,
                multimask_output=False
            )

        # Visualize
        figs = show_masks(image, masks, scores, point_coords=point_coords, box_coords=box_xyxy, input_labels=point_labels, borders=True)
        plt.show()
        plt.close(figs[0][0])

        figs = show_masks(image, masks, scores, borders=True)
        plt.show()
        plt.close(figs[0][0])

@dataclass
class SAMRegionPredictorConfig:
    use_positive_points: bool = True
    use_negative_points: bool = True
    use_bbox: bool = True

    filter_low_confidence_masks: bool = True
    min_mask_score: float = 0.4

    # Sampling parameters
    num_positive: int = 10
    num_negative: int = 10
    min_dist_from_mask: int = 100
    bias_positive: bool = True
    bias_negative: bool = False

@dataclass
class SAMRegionPredictorResult:
    masks: np.ndarray = None
    scores: np.ndarray = None
    logits: np.ndarray = None
    point_coords: np.ndarray = None
    point_labels: np.ndarray = None
    box: np.ndarray = None

class SAMRegionPredictor(torch.nn.Module):
    def __init__(self, config: SAMRegionPredictorConfig):
        super().__init__()
        self.config = config
        self.predictor = SAM2ImagePredictor.from_pretrained('facebook/sam2-hiera-large')

    def predict(self, mask: np.ndarray, image: PILImage) -> list[SAMRegionPredictorResult]:
        components = extract_components(mask)  # (n, h, w)

        # TODO can potentially batch all components for the SAMRegionPredictor
        # see https://colab.research.google.com/github/facebookresearch/sam2/blob/main/notebooks/image_predictor_example.ipynb

        results = []
        for component in components:
            # Construct point inputs
            point_coords = None
            point_labels = None

            if self.config.use_positive_points or self.config.use_negative_points:
                point_coords = []
                point_labels = []

                pos_points, neg_points = sample_points(
                    component,
                    num_positive=self.config.num_positive,
                    num_negative=self.config.num_negative,
                    min_dist_from_mask=self.config.min_dist_from_mask,
                    bias_positive=self.config.bias_positive,
                    bias_negative=self.config.bias_negative
                )

                if self.config.use_positive_points:
                    pos_points = pos_points[:, ::-1]  # (n, 2) from row, col (y,x) --> (x,y)
                    point_coords.append(pos_points)
                    point_labels.extend([1] * pos_points.shape[0])

                if self.config.use_negative_points:
                    neg_points = neg_points[:, ::-1]
                    point_coords.append(neg_points)
                    point_labels.extend([0] * neg_points.shape[0])

                point_coords = np.concatenate(point_coords, axis=0) # (n, 2)
                point_labels = np.array(point_labels)

            # Construct bbox input
            box_xyxy = None
            if self.config.use_bbox:
                box_xyxy = bbox_from_mask(component[None, ...]).squeeze()  # (4,) in xyxy

            # Get prediction
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                self.predictor.set_image(image)
                masks, scores, logits = self.predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=box_xyxy,
                    multimask_output=False
                )

            if not self.config.filter_low_confidence_masks or scores[0] >= self.config.min_mask_score:
                results.append(
                    SAMRegionPredictorResult(
                        masks=masks, # (n_masks, h, w)
                        scores=scores, # (n_masks,)
                        logits=logits,
                        point_coords=point_coords, # (n_points, 2)
                        point_labels=point_labels, # (n_points,)
                        box=box_xyxy # (4,) in xyxy
                    )
                )

        return results

def get_rle_dict(mask: np.ndarray, image_path: str) -> dict:
    mask = mask.squeeze()
    rle_dict = mask_utils.encode(np.asfortranarray(mask).astype(np.uint8))
    rle_dict['counts']= rle_dict['counts'].decode('utf-8')
    rle_dict['image_path'] = image_path

    return rle_dict

def save_figure(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)

def parse_args(cl_args: list[str] = None, config_str: str = None):
    parser = ArgumentParser()
    parser.add_argument('--rle_input_dir', type=str, required=True)
    parser.add_argument('--rle_output_dir', type=str, required=True)
    parser.add_argument('--vis_output_dir', type=str, default=None)

    parser.add_argument('--masks.intersect_with_foreground', type=bool, default=True)
    parser.add_argument('--masks.rembg_model_name', type=str, choices=['u2net', 'sam_prompt'])
    parser.add_argument('--masks.region_predictor_config', type=SAMRegionPredictorConfig, default=SAMRegionPredictorConfig())

    parser.add_argument('--device', type=str, default='cuda')

    if config_str:
        args = parser.parse_string(config_str)
    else:
        args = parser.parse_args(cl_args)

    return args, parser

def main(cl_args: list[str] = None, config_str: str = None):
    args, parser = parse_args(cl_args, config_str)

    # Dump config to both output directories
    os.makedirs(args.rle_output_dir, exist_ok=True)
    parser.save(args, os.path.join(args.rle_output_dir, 'config.yaml'), overwrite=True)

    if args.vis_output_dir:
        os.makedirs(args.vis_output_dir, exist_ok=True)
        parser.save(args, os.path.join(args.vis_output_dir, 'config.yaml'), overwrite=True)

    region_predictor = SAMRegionPredictor(args.masks.region_predictor_config).to(args.device)
    if args.masks.intersect_with_foreground:
        rembg_session = new_session(model_name=args.masks.rembg_model_name)

    # Process RLEs
    rle_input_paths = list_paths(args.rle_input_dir, exts=['.json'])
    for rle_input_path in tqdm(rle_input_paths):
        rle_output_path = rle_input_path.replace(args.rle_input_dir, args.rle_output_dir)
        if os.path.exists(rle_output_path):
            logger.debug(f'Skipping {rle_input_path}')
            continue

        logger.debug(f'Processing {rle_input_path}')
        mask, image_path = load_mask(rle_input_path)
        image = open_image(image_path)

        predictions = region_predictor.predict(mask, image)

        if predictions != []:
            mask_union = np.concatenate([r.masks for r in predictions]).any(axis=0) # Union of all masks (h, w)
        else:
            mask_union = np.zeros_like(mask) # (h, w)

        if args.masks.intersect_with_foreground:
            foreground_mask = np.array(remove(image, session=rembg_session, post_process_mask=True, only_mask=True)).astype(bool) # (h, w)
            mask_union = np.logical_and(mask_union, foreground_mask)

        # Output mask rle
        rle_dict = get_rle_dict(mask_union, rle_output_path)

        os.makedirs(os.path.dirname(rle_output_path), exist_ok=True)
        with open(rle_output_path, 'w') as f:
            json.dump(rle_dict, f)

        # Output visualization
        if args.vis_output_dir:
            # Visualize components individually
            for i, prediction in enumerate(predictions):
                figs = show_masks(image, prediction.masks, prediction.scores, point_coords=prediction.point_coords, box_coords=prediction.box, input_labels=prediction.point_labels, borders=True)
                fig_path = rle_input_path.replace(args.rle_input_dir, args.vis_output_dir).replace('.json', f'_{i}.jpg')
                save_figure(figs[0][0], fig_path)

            # Visualize foreground
            if args.masks.intersect_with_foreground:
                figs = show_masks(image, [foreground_mask], [1.], borders=True)
                fig_path = rle_input_path.replace(args.rle_input_dir, args.vis_output_dir).replace('.json', '_foreground.jpg')
                save_figure(figs[0][0], fig_path)

            # Visualize union
            figs = show_masks(image, [np.ascontiguousarray(mask_union)], [1.], borders=True)
            fig_path = rle_input_path.replace(args.rle_input_dir, args.vis_output_dir).replace('.json', '_union.jpg')
            save_figure(figs[0][0], fig_path)

# %%
if __name__ == '__main__':
    import coloredlogs
    coloredlogs.install(level='DEBUG')
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('sam2').setLevel(logging.WARNING)

    main(config_str='''
        rle_input_dir: /shared/nas2/blume5/sp25/partonomy/partonomy_private/data/partonomy_part_heatmaps

        rle_output_dir: /shared/nas2/blume5/sp25/partonomy/partonomy_private/data/partonomy_sam_predictions
        # rle_output_dir: /shared/nas2/blume5/sp25/partonomy/partonomy_private/data/partonomy_sam_predictions-no_bbox

        vis_output_dir: /shared/nas2/blume5/sp25/partonomy/results/2025_02_25-00_45_00-sam_visualizations
        # vis_output_dir: /shared/nas2/blume5/sp25/partonomy/results/2025_02_25-00_45_00-sam_visualizations-no_bbox

        masks:
            region_predictor_config:
                use_positive_points: false
                use_negative_points: false
                use_bbox: true

                filter_low_confidence_masks: true
                min_mask_score: 0.4

                num_positive: 2
                num_negative: 50
                min_dist_from_mask: 5
                bias_positive: true
                bias_negative: false

            intersect_with_foreground: true
            rembg_model_name: u2net

        device: cuda:0
    ''')

    # TODO use proba-.5 heatmaps
    # TODO connect very close connected components (within a certain threshold)
    # TODO try sampling negative points where no connected component is positive
    # TODO script to merge SAM predictions with human annotations, preferring human annotations (where provided)

    # TODO use semantic-sam for region proposals, and select the best one?

    # TODO Use distribution of part counts per image to determine whether to merge blobs or process individually (see count_parts_per_image.py)

    # TODO See how many concepts actually have multiple annotations of the same part in a single image to see how widespread the problem is