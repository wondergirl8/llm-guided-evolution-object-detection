"""MMDetection Faster R-CNN built only from deterministic random weights."""
import hashlib
from pathlib import Path
import torch
from .validator import validate_genes

def _remove_init_cfg(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "init_cfg": value[key] = None
            else: _remove_init_cfg(child)
    elif isinstance(value, (list, tuple)):
        for child in value: _remove_init_cfg(child)

def build_config(genes):
    validate_genes(genes)
    import mmdet
    from mmengine.config import Config
    cfg = Config.fromfile(str(Path(mmdet.__file__).parent / ".mim/configs/_base_/models/faster-rcnn_r50_fpn.py")).model
    _remove_init_cfg(cfg)
    cfg.data_preprocessor.bgr_to_rgb = False
    width = genes["fpn_channels"]
    cfg.neck.out_channels = width
    cfg.rpn_head.in_channels = width
    cfg.rpn_head.feat_channels = genes["rpn_channels"]
    cfg.rpn_head.num_convs = genes["rpn_convs"]
    cfg.rpn_head.anchor_generator.scales = genes["anchor_scales"]
    cfg.rpn_head.anchor_generator.ratios = genes["anchor_ratios"]
    cfg.roi_head.bbox_roi_extractor.out_channels = width
    cfg.roi_head.bbox_roi_extractor.roi_layer.output_size = genes["roi_size"]
    cfg.roi_head.bbox_roi_extractor.roi_layer.sampling_ratio = genes["roi_sampling_ratio"]
    cfg.roi_head.bbox_head.in_channels = width
    cfg.roi_head.bbox_head.fc_out_channels = genes["box_fc_channels"]
    cfg.roi_head.bbox_head.roi_feat_size = genes["roi_size"]
    cfg.roi_head.bbox_head.num_classes = 1
    return cfg

def state_sha256(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode()); digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode()); digest.update(value.numpy().tobytes())
    return digest.hexdigest()

def build_detector(cfg, *, seed):
    from mmdet.registry import MODELS
    from mmdet.utils import register_all_modules
    register_all_modules()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = MODELS.build(cfg)
        for module in model.modules():
            if hasattr(module, "init_cfg"): module.init_cfg = None
        model.init_weights()
    return model, {"policy":"deterministic_framework_random_initialization","seed":seed,
                   "pretrained":False,"checkpoint":None,"initial_state_sha256":state_sha256(model)}
