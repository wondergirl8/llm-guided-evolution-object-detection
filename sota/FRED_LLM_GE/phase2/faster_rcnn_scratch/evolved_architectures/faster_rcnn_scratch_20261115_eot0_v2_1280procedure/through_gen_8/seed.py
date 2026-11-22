"""Faster R-CNN V2: bounded model and input-resolution genes."""
def build_model_config():
    from sota.FRED_LLM_GE.phase2.faster_rcnn_scratch.v2.genome import config_from_genes
    return config_from_genes({**FeatureGenes.values, **ProposalGenes.values,
                              **RoIGenes.values, **InputGenes.values})

# --OPTION--
class FeatureGenes:
    values = {'fpn_channels': 256}

# --OPTION--
class ProposalGenes:
    values = {'rpn_channels': 256, 'rpn_convs': 1, 'anchor_scales': [8], 'anchor_ratios': [0.5, 1.0, 2.0]}

# --OPTION--
class RoIGenes:
    values = {'roi_size': 7, 'roi_sampling_ratio': 0, 'box_fc_channels': 1024}

# --OPTION--
class InputGenes:
    values = {'img_width': 1280}

