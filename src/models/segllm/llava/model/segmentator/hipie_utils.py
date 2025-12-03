from detectron2.projects.hipie import add_hipie_config,HIPIE_IMG
from argparse import Namespace
import torch
from detectron2.config import get_cfg
from detectron2.modeling import build_model
import torch.nn.functional as F
import numpy as np
from segment_anything.utils.transforms import ResizeLongestSide
from torch import nn
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.checkpoint import DetectionCheckpointer

from detectron2.projects.hipie.util.misc import NestedTensor, interpolate, nested_tensor_from_tensor_list, inverse_sigmoid
def forward_text_hack(self,captions,device,task=None):
    hidden = torch.stack(captions)
    # print(hidden.shape)
    # print(masks.shape)
    return {
        "hidden":hidden.to(device), # 1 512 768
        "masks": torch.ones(hidden.shape[0],hidden.shape[1]).to(device), # 1 512
    }
    
def calculate_iou(boxes1, boxes2):
    """
    Calculate the Intersection over Union (IoU) for each pair of bounding boxes.
    
    Args:
    boxes1: torch.Tensor of shape (N, 4), where each row is (x1, y1, x2, y2)
    boxes2: torch.Tensor of shape (N, 4), where each row is (x1, y1, x2, y2)
    
    Returns:
    torch.Tensor of shape (N,) with IoU for each pair of bounding boxes.
    """
    boxes1 = boxes1.detach().cpu()
    boxes2 = boxes2.detach().cpu()
    # Calculate the intersection coordinates
    inter_x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
    inter_y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
    inter_x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
    inter_y2 = torch.min(boxes1[:, 3], boxes2[:, 3])
    
    # Calculate the intersection area
    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    
    # Calculate the area of each bounding box
    boxes1_area = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    boxes2_area = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # Calculate the union area
    union_area = boxes1_area + boxes2_area - inter_area
    
    # Calculate the IoU
    iou = inter_area / union_area
    
    return iou
HIPIE_IMG.forward_text = forward_text_hack
def setup_cfg(args,weight,device=None):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_hipie_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # HIPIE
    if args.task == "grounding":
        cfg.DATASETS.TEST = ("refcoco-unc-val", )
    elif args.task == "detection":
        cfg.DATASETS.TEST = ("coco_2017_val", )
    cfg.MODEL.WEIGHTS = weight
    if device is not None:
        cfg.MODEL.DEVICE = device 
        
    cfg.MODEL.MASKDINO.ENABLED = False
    #cfg.freeze()
    return cfg

class Preprocessor:


    def __init__(self,img_size,transform):
        self.img_size = img_size
        self.transform = transform
    
    
    def pad_only(self,x):
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
    
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # No normalization
        return self.pad_only(x)
    
    def __call__( self,image,masks=None):
        # expects masks to be a list of masks: [ref, gt] (or just [gt])
        input_image = self.transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        input_size = tuple(input_image_torch.shape[-2:])
        input_image = self.preprocess(input_image_torch)
        if masks is not None:
            masks = [self.transform.apply_image(mask) for mask in masks]
            masks = torch.as_tensor(np.stack(masks))
            masks = masks[:,None]
            #mask = masku..permute(2, 0, 1).contiguous()[None, :, :, :]
            masks = self.pad_only(masks)
            assert masks.shape[-2:] == input_image.shape[-2:]
        return dict(
            image=input_image.squeeze(0), # C HW 
            input_size=input_size,
            mask=masks[-1], # C HW          if only masks=[gt] is passed in, mask = aux_mask = gt mask 
            aux_mask=masks[0], # C HW 
        )
       
def hasNan(output):
    if isinstance(output, torch.Tensor):
        outputs = torch.isnan(output).any()
    if isinstance(output, list):
        outputs = any([hasNan(x) for x in output])
    elif isinstance(output, dict):
        outputs = any([hasNan(x) for x in output.values()])
    elif isinstance(output,tuple):
        outputs = any([hasNan(x) for x in output])
    elif isinstance(output,NestedTensor):
        outputs = hasNan([output.tensors,output.mask])
    else:
        outputs = False
    return outputs
def nan_hook(self, inp, output):
    if hasNan(output):
        print("In", self.__class__.__name__)
        raise RuntimeError(f"Found NAN in output at {self.__class__.__name__}")

class HIPIESementator(nn.Module):
    
    def __init__(self, 
        config_file = './llava/model/segmentator/hipie_configs/training/r50_pretrain.yaml',
        weight = './pretrained_weights/hipie/r50_parts.pth',
        prompt_dim=256,
        dice_weight=5.0,
        mask_weight=5.0,
        num_tokens=1,
        detection_loss=False,
        weight_ref=0.5,
        weight_tgt=1.0,
        sparse_embed_L2_loss_weight=1e-3,
        max_rounds=3,
        val_batch_size=1024,
        use_norm=False,
        **kwargs
        ) -> None:
        super().__init__()
        args = Namespace()
        self.max_rounds = max_rounds
        self.val_batch_size = val_batch_size
        self.img_size = 1024
        self.prompt_dim = 768
        self.output_token_dim = 768 * 8
        args.config_file = config_file
        args.opts = ['OUTPUT_DIR','outputs/test_r50_maskdino_pan_fixed_lan','MODEL.MASKDINO.CONFIG_PATH', 'llava/model/segmentator/hipie_configs/mask_dino/maskdino_R50_bs16_50ep_3s_dowsample1_2048.yaml']
        args.task = "detection"
        cfg = setup_cfg(args,weight,device='cuda')
        model = build_model(cfg)
        checkpointer = DetectionCheckpointer(model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.sparse_embed_L2_loss_weight = sparse_embed_L2_loss_weight

        self.hipie = model
        self.transform = ResizeLongestSide(1024)
        self.process_images = Preprocessor(
            self.img_size,
            self.transform

        )
        if use_norm:
            self.norm = nn.LayerNorm(self.prompt_dim)
        else:
            self.norm = nn.Identity()
            
        for submodule in self.hipie.modules():
            submodule.register_forward_hook(nan_hook)
        
    def forward(self,images,prompts,targets,prev_masks,sample_indices=None):
        return self.segment_mask(images,prompts,targets=targets,prev_masks=prev_masks,sample_indices=sample_indices)
        
    # def requires_grad_(self, requires_grad: bool = True):
    #     if requires_grad:
    #         self.sam.image_encoder.requires_grad_(True)
    #     else:
    #         self.sam.image_encoder.requires_grad_(False)
    #     return self
    def forward_features(self,samples):
        image_sizes = samples.image_sizes
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples, size_divisibility=32)
        with torch.autocast(device_type='cuda', enabled=False, dtype=torch.bfloat16):
            features, pos = self.hipie.detr.detr.backbone(samples.to(torch.bfloat16))
        return features, pos
    
    def requires_grad_(self, requires_grad: bool = True):
        if requires_grad:
            self.hipie.image_encoder.requires_grad_(True)
        else:
            self.sam.image_encoder.requires_grad_(False)
        return self


    def segment_mask(
        self,
        images,
        sparse_embeddings,
        original_size=None,
        targets=None,
        prev_masks=None,
        sample_indices=None
    ) -> None:
        '''
        INPUT ARGS          (B = num images/convs, L = num masks in THIS image/conv)
        images:             List[]: (B) x 3 x 1024 x 1024                  
        sparse_embeddings:  List[]: (B) x L X D         ([SEG] token hidden states, D=256 or 512)
        targets:            List[]: (B) x L x 1 x 1024x1024  (GT target masks) 
        prev_masks:         List[]: (B) x L x 1 x 1024  (GT reference masks) 
        '''
        
        ## prepare_inputs
        n = len(images)
        batched_inputs = []
        # normalize images
        hipie = self.hipie
        images = [hipie.normalizer(x) for x in images]
        images = ImageList.from_tensors(images)
        features, pos =  self.forward_features(images)
        img_indices = []
        selected = []
        
        for i,x in enumerate(targets):
            n = len(targets[i])
            img_indices.extend([i]*n)
            if n > self.max_rounds:
                selected_local = [True]*self.max_rounds + [False] * (n-self.max_rounds)
                np.random.shuffle(selected_local)
                selected.extend(selected_local)
            else:
                selected.extend([True]*n)
        sparse_embeddings = torch.cat(sparse_embeddings)
        
        targets = torch.cat(targets)
        if not (targets.sum((-1,-2)) > 0).all():
            print("** WARNING: (segllm:hipie_utils) targets has no positive pixels : ", targets.sum((-1,-2)))

        prev_masks = torch.cat(prev_masks)
        img_indices = np.array(img_indices)
        if self.training:
            targets = targets[selected]
            prev_masks =prev_masks[selected]
            img_indices = img_indices[selected]
            sparse_embeddings = sparse_embeddings[selected]
        #sparse_embeddings = torch.nan_to_num(sparse_embeddings,0)
        assert not sparse_embeddings.isnan().any()
        assert not targets.isnan().any()
        assert not prev_masks.isnan().any()
        # assert not img_indices.isnan().any()
        has_previous_mask = prev_masks.sum((-1,-2))>0
        has_target_mask = targets.sum((-1,-2))>0
        # assert has_target_mask.all()
        if not has_target_mask.all():
            print("** WARNING: (segllm:hipie_utils) targets has no positive pixels : ", targets.sum((-1,-2)))
        raw_image_sizes = images.image_sizes
        processed_image_sizes = []
        for i in range(len(features)):
            features[i].tensors = features[i].tensors[img_indices]
            features[i].mask = features[i].mask[img_indices]
        for j in img_indices:
            processed_image_sizes.append(raw_image_sizes[j])
        pos = list([x[img_indices] for x in pos])
        # build instances
        idx = 0
        n_mask = len(targets)
        all_tgts = torch.cat([prev_masks,targets],dim=1) # n_mask * 2 * 1024 * 1024
        
        sparse_embeddings = sparse_embeddings.view(n_mask,-1,768) # n_mask x 16 x 768
        sparse_embeddings = self.norm(sparse_embeddings) # norm will be idenity if not enabled 
        seq_len = sparse_embeddings.shape[1]
        half_len = seq_len//2
        batched_inputs = []
        seq_indices = np.arange(seq_len)
        if self.training:
            for i in range(n_mask):
                instances = Instances((1024,1024))
                if has_previous_mask[i]:
                    instances.gt_classes = torch.tensor([1,2], dtype=torch.int64) # two classes
                    instances.gt_masks = BitMasks(all_tgts[i])
                    instances.gt_boxes = instances.gt_masks.get_bounding_boxes() 
                    instances.is_thing = torch.tensor([True,True])
                    positive_map_label_to_token =  {1: seq_indices[:half_len],2:seq_indices[half_len:]}
                    positive_maps = torch.zeros((2, seq_len), dtype=torch.float)
                    positive_maps[0,:half_len] = 1
                    positive_maps[1,half_len:] = 1
                    instances.positive_map = positive_maps.bool()
                    batched_inputs.append(
                        dict(
                            instances = instances,
                            expressions = sparse_embeddings[i]
                        )
                    )
                else:
                    instances.gt_classes = torch.tensor([1], dtype=torch.int64) # two classes
                    instances.gt_masks = BitMasks(all_tgts[i,1:]) # n x 1 x 1024 x 1024
                    instances.gt_boxes = instances.gt_masks.get_bounding_boxes() 
                    instances.is_thing = torch.tensor([True])
                    positive_map_label_to_token =  {1: seq_indices[:half_len],2:seq_indices[half_len:]}
                    positive_maps = torch.zeros((1, seq_len), dtype=torch.float)
                    positive_maps[0,half_len:] = 1
                    instances.positive_map = positive_maps.bool()
                    batched_inputs.append(
                        dict(
                            instances = instances,
                            expressions = sparse_embeddings[i]
                        )
                    )
            device = sparse_embeddings.device
            gt_instances = [x["instances"].to(device) for x in batched_inputs]
            detr_targets = hipie.prepare_targets(gt_instances)
            captions = [x["expressions"] for x in batched_inputs]
            language_dict_features = hipie.forward_text(captions, device=device,  task='detection')
            language_dict_features['hidden'] = language_dict_features['hidden'].to(torch.bfloat16)
            #language_dict_features['m'] = language_dict_features['hidden'].to(torch.bfloat16)
            weight_dict = hipie.criterion.weight_dict
            
            output, loss_dict = hipie.detr.coco_forward(None, detr_targets, hipie.criterion, train=True, language_dict_features=language_dict_features, task='detection',
                                                        features=features,pos=pos,image_sizes=[(1024,1024)]*n_mask)
            
            for k in loss_dict.keys():
                if hipie.detr.decouple_decoder and '_maskdino' in k: # hack, do not drop mask dino loss
                    continue
                if k in weight_dict:
                        loss_dict[k] *= (weight_dict[k] * hipie.loss_weight_det)
            pred_masks = output['out_fg']['pred_masks']
            pred_masks = torch.stack([x[0,-1] for x in pred_masks ])
            pred_logits = output['out_fg']['pred_logits'] # n x 900 x 16
            with torch.no_grad():
                pred_masks = F.interpolate(pred_masks,(1024,1024),mode='bilinear')
                pred_masks = pred_masks > 0
                targets = targets.bool()
                intersection = (pred_masks & targets).sum()
                union = (pred_masks | targets).sum()
                iou = intersection / (union+1e-4)
            losses = sum(loss_dict.values())
            loss_dict['total_loss'] = losses + (sparse_embeddings**2).mean() * self.sparse_embed_L2_loss_weight
            loss_dict['iou_train'] = iou
            return None,loss_dict
            
        else:
            device = sparse_embeddings.device
            self.hipie.eval()
            captions = [sparse_embeddings[i] for i in range(len(sparse_embeddings))]
            positive_map_label_to_token =  {1: seq_indices[:half_len],2:seq_indices[half_len:]}
            num_classes = len(positive_map_label_to_token)
            language_dict_features = hipie.forward_text(captions, device=device,  task='detection')
            language_dict_features['hidden'] = language_dict_features['hidden'].to(torch.bfloat16)
            is_thing =  [{1: True,2:True}]*len(captions)
            
            max_inference_bs = self.val_batch_size
            start = 0
            box_cls = []
            box_pred = []
            mask_pred = []
            iou_pred = []
            while start < n_mask:
                end = start + max_inference_bs
                with torch.no_grad():
                    local_features = list([
                        NestedTensor(x.tensors[start:end],x.mask[start:end]) for x in features
                    ])
                    local_pos = list([x[start:end] for x in pos])
                    local_lang_dict = dict(
                                                                        hidden=language_dict_features['hidden'][start:end],
                                                                        masks=language_dict_features['masks'][start:end],
                                                                    )
                    output, loss_dict = hipie.detr.coco_inference(None,
                                                                  None, 
                                                                  hipie.criterion,
                                                                  train=False,
                                                                  language_dict_features=local_lang_dict,
                                                                  task='detection',
                                                                bg_queries_lang=None,
                                                                features=local_features,pos=local_pos,image_sizes=[(1024,1024)]*len(local_lang_dict['hidden']))
                    box_cls.append(output["pred_logits"])
                    box_pred.append(output["pred_boxes"])
                    mask_pred.append(output["pred_masks"] )
                    if hipie.detr.use_iou_branch:
                        iou_pred.append(output["pred_boxious"])
                    else:
                        iou_pred = [None]
                start += max_inference_bs
            box_cls = torch.cat(box_cls)
            box_pred = torch.cat(box_pred)
            mask_pred = torch.cat(mask_pred)
            image_sizes=[(1024,1024)]*n_mask
            if hipie.detr.use_iou_branch:
                iou_pred = torch.cat(iou_pred)
            else:
                iou_pred = [None]
            with torch.no_grad():
                results = hipie.inference(box_cls, box_pred, mask_pred, processed_image_sizes, positive_map_label_to_token, num_classes, task='detection', iou_pred=iou_pred,is_thing=is_thing,sizes=image_sizes,output=output,
                                        bg_queries_lang=None,test_labels=None,images=None)
            all_masks = []
            all_boxes = []
            for i in range(n_mask):
                try:
                    pred_masks = results[i]['instances'].pred_masks #N_PRED, 1, 1024,1024
                except:
                    breakpoint()
                pred_boxes = results[i]['instances'].pred_boxes #N_PRED, 1, 1024,1024
                scores = results[i]['instances'].scores #N_PRED, 1,
                labels = results[i]['instances'].pred_classes
                # here 1,2 -> 0, 1
                scores[labels == 0] = -1
                best_mask_idx = scores.argmax().item()
                all_masks.append(pred_masks[best_mask_idx])
                all_boxes.append(pred_boxes[best_mask_idx])
            all_masks = torch.cat(all_masks) #N x 1024 x 1024
            targets = targets.bool()[:,0]
            intersection = (all_masks & targets).sum((-1,-2))
            union = (all_masks | targets).sum((-1,-2))
            intersection = (all_masks & targets).sum((-1,-2))
            iou_per_image = (intersection / (union + 1e-4))
            tgt_bbox =  BitMasks(targets).get_bounding_boxes()
            tgt_bbox = tgt_bbox.tensor
            all_boxes = torch.cat([x.tensor for x in all_boxes]).detach()
            iou_boxes = calculate_iou(all_boxes,tgt_bbox)
            # breakpoint()
            assert len(sample_indices) == len(img_indices)
            payload = {
                "union":union.sum().item(),                                 # N
                "intersection":intersection.sum().item(),                   # N
                "img_indices":sample_indices,#img_indices,   # N
                "inter_per_mask":intersection.detach().cpu().numpy(),       # N
                "union_per_mask":union.detach().cpu().numpy(),              # N
                "iou_per_mask":iou_per_image.detach().cpu().numpy(),        # N
                "mask":all_masks.detach().cpu().numpy(),                    # N x 1024 x 1024
                "boxes":all_boxes.detach().cpu().numpy(),                   # N x 4
                "detections_loss":0.0,
                "box_ious":iou_boxes.detach().cpu().numpy(),                # N
                "train_acc":0.0,
                "train_mIoU":0.0,
                "loss_dice":0.0,
                "loss_mask":0.0,
                "total_loss":0.0,               
            }
            return {},payload
        '''
        N = num segmentation mask = sum([num_rounds(conv) for conv in batch])
        img_indices = [0,0,1,1]
        each image correspond to 1 conversation
        img_indices[0] = 0 means all_masks[0] is the 1st round output for 1st conversation
        img_indices[1] = 0 means all_masks[1] is the 2nd round output for 1st conversation
        '''
            

        
            
        

if __name__ == "__main__":
    config_file = './llava/model/segmentator/hipie_configs/training/r50_pretrain.yaml'
    weight = './pretrained_weights/hipie/r50_parts.pth'
    args = Namespace()
    args.config_file = config_file
    args.opts = ['OUTPUT_DIR','outputs/test_r50_maskdino_pan_fixed_lan','MODEL.MASKDINO.CONFIG_PATH', 'llava/model/segmentator/hipie_configs/mask_dino/maskdino_R50_bs16_50ep_3s_dowsample1_2048.yaml']
    args.task = "detection"
    cfg = setup_cfg(args,weight,device='cuda')
    breakpoint()
    model = build_model(cfg)
    
    instances = Instances((1024,1024))
    instances.gt_classes = torch.tensor([1,2,3,1], dtype=torch.int64)
    n = len(instances)
    instances.gt_masks = BitMasks(torch.rand((n, 1024, 1024)))
    instances.gt_boxes = instances.gt_masks.get_bounding_boxes() 
    instances.is_thing = torch.tensor([True,]*n)
    positive_map_label_to_token =  {1: [0,1,2],2:[3,4,5],3:[4,5]}
    positive_maps = torch.zeros((len(instances), 6), dtype=torch.float)
    for i in range(len(instances)):
        lablel = instances.gt_classes[i].item()
        positive_maps[i,positive_map_label_to_token[lablel]] = 1.0

    instances.positive_map = positive_maps.bool()
    batched_inputs  = [
        {
            "image":torch.rand((3,1024,1024)).cuda(),
            "instances":instances,
            "task":"detection",
            "positive_map_label_to_token":positive_map_label_to_token,
            "positive_map": positive_maps,
            "expressions": torch.rand(6,768),
            "is_thing":{
                1:True,
                2:True,
                3:True
            }
        }
    ]
    model.train()
    with torch.no_grad():
        output = model(batched_inputs)
    
    model.eval()
    with torch.no_grad():
        output_eval = model(batched_inputs)  
    model.eval()
