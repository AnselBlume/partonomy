import yaml
import pathlib
CURR_PATH = pathlib.Path(__file__).parent.resolve()
import importlib
def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

import torch
def build_segmentator(config):
    if '0' in config:
        config = config.replace('0', '')
    config_file = CURR_PATH / f'{config}.yaml'
    with open(config_file) as f:
        config = yaml.safe_load(f.read())
    model_config  =  config['model']
    model = instantiate_from_config(model_config)
    ckpt = config['ckpt']
    if ckpt:
        sd = torch.load(ckpt,map_location='cpu')["state_dict"]
        model.load_state_dict(sd, strict=False)
    # if modality == 'image':
    #     if config.get("processor") == 'clip':
    #         processor = ClipVisionPreprocessorForLLM(pretrained=config['model']['params']['pretrained'])
    #     else:
    #         processor = ImageVAEProcesser(config['image_size'])
    # elif modality == 'audio':
    #     if config.get("processor") == 'npz':
    #         processor = NpzLoader()
    #     else:
    #         processor = AudioVAEProcesser(config['data']['params']['sample_rate'])
    return model
