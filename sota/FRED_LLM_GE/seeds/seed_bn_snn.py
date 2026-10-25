"""Experimental raw-event BN-SNN genome; protected dynamics live outside this file."""
from sota.FRED_LLM_GE.phase1_detection.seeds.bn_snn.network import BNSNN


def build_model(num_classes=1):
    # The SSD classifier includes background (0); drone is internal class 1.
    return BNSNN(num_classes=num_classes + 1, base_channels=BASE_CHANNELS,
                 stage_depths=STAGE_DEPTHS)

# --OPTION--
BASE_CHANNELS = 32
# --OPTION--
STAGE_DEPTHS = (2, 2, 2, 2)
