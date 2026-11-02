"""MMDetection R50-FPN prototype genome. Only literal gene values are mutable.

The trainer validates this source before execution. The protected builder owns
all data, loss, initialization, and evaluation policies.
"""


def build_model_config():
    from sota.FRED_LLM_GE.phase1_detection.seeds.faster_rcnn.network import build_config
    return build_config({**FeatureGenes.values, **ProposalGenes.values, **RoIGenes.values})


# --OPTION--
class FeatureGenes:
    values = {"fpn_channels": 256}


# --OPTION--
class ProposalGenes:
    values = {"rpn_channels": 256, "rpn_convs": 1,
              "anchor_scales": [8], "anchor_ratios": [0.5, 1.0, 2.0]}


# --OPTION--
class RoIGenes:
    values = {"roi_size": 7, "roi_sampling_ratio": 0, "box_fc_channels": 1024}
