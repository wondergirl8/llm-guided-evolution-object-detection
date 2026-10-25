"""Experimental EMS-ResNet34-style YOLO seed; dynamics and training are protected."""
from sota.FRED_LLM_GE.phase1_detection.seeds.ems_yolo.network import EMSYolo


def build_model(num_classes=1):
    return EMSYolo(num_classes=num_classes, base_channels=BASE_CHANNELS,
                   stage_depths=STAGE_DEPTHS, neck_channels=NECK_CHANNELS)

# --OPTION--
BASE_CHANNELS = 64
# --OPTION--
STAGE_DEPTHS = (3, 4, 6, 3)
# --OPTION--
NECK_CHANNELS = 256
