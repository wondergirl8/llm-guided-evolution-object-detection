"""Protected Faster R-CNN model builder and explicit checkpoint transfer."""
from pathlib import Path
import hashlib
import torch
from .validator import validate_genes

CHECKPOINT_URL = ("https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/"
                  "faster_rcnn_r50_fpn_1x_coco/"
                  "faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth")


def build_config(genes):
    validate_genes(genes)
    import mmdet
    from mmengine.config import Config
    path = Path(mmdet.__file__).parent / ".mim/configs/_base_/models/faster-rcnn_r50_fpn.py"
    cfg = Config.fromfile(str(path)).model
    cfg.backbone.init_cfg = None  # Initialization is exclusively owned below.
    cfg.data_preprocessor.bgr_to_rgb = False  # FRED/PIL already provides RGB.
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


def build_detector(cfg, checkpoint):
    from mmdet.registry import MODELS
    from mmdet.utils import register_all_modules
    register_all_modules()
    model = MODELS.build(cfg)
    model.init_weights()
    path = Path(checkpoint)
    # Download is an explicit setup step, never a candidate side effect.
    if not path.is_file():
        raise FileNotFoundError(f"Required COCO checkpoint not found: {path}")
    state = torch.load(path, map_location="cpu")["state_dict"]
    own = model.state_dict()
    compatible = {key: value for key, value in state.items()
                  if key in own and own[key].shape == value.shape}
    skipped = {key: {"checkpoint_shape": list(value.shape),
                     "model_shape": list(own[key].shape) if key in own else None}
               for key, value in state.items() if key not in compatible}
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    if not any(key.startswith("backbone.") for key in compatible):
        raise ValueError("Checkpoint has no compatible backbone weights")
    return model, {"checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                   "checkpoint_url": CHECKPOINT_URL, "loaded": sorted(compatible),
                   "skipped": skipped, "initialized": list(missing),
                   "unexpected": list(unexpected)}
