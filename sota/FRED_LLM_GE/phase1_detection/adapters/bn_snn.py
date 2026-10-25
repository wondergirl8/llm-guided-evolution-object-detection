"""Raw temporal-event adapter; independent of the LLM and controller."""
import random

import numpy as np
import torch
from torchvision.ops import nms

from .base import FamilyAdapter
from ..bn_snn_search_space import build, built_architecture
from ..seeds.bn_snn.detection import anchors_for, decode, multibox_loss


class BNSNNAdapter(FamilyAdapter):
    name = 'bn_snn'

    def __init__(self, settings, fred_dir):
        super().__init__(settings, fred_dir)
        recipe = settings['training']
        if (recipe['optimizer'] != 'AdamW' or recipe['amp'] is not False
                or recipe['augmentation'] != 'none'):
            raise ValueError('BN-SNN supports AdamW, full precision, and no augmentation')
        if (recipe['lr'] <= 0 or recipe['weight_decay'] < 0 or recipe['grad_clip'] <= 0
                or type(recipe['microbatch_size']) is not int or recipe['microbatch_size'] < 1):
            raise ValueError('invalid BN-SNN optimizer or microbatch settings')
        if len(settings['anchor_sizes']) != 3 or any(s <= 0 for s in settings['anchor_sizes']):
            raise ValueError('require three positive SSD anchor sizes')
        if settings['pre_nms_topk'] < 1 or settings['max_detections'] < 1:
            raise ValueError('detection limits must be positive')

    def build(self, variant_path, num_classes, device):
        built_architecture(variant_path)
        return build(variant_path, num_classes).to(device)

    def train(self, model, data, proxy, device):
        recipe = self.settings['training']
        random.seed(proxy['seed'])
        np.random.seed(proxy['seed'])
        generator = torch.Generator().manual_seed(proxy['seed'])
        optimizer = torch.optim.AdamW(model.parameters(), lr=recipe['lr'],
                                     weight_decay=recipe['weight_decay'])
        batch_size = proxy['batch_size']
        micro = recipe['microbatch_size']
        model.train()
        for epoch in range(proxy['epochs']):
            order = torch.randperm(len(data['images']), generator=generator)
            total_loss = 0.0
            for group in order.split(batch_size):
                optimizer.zero_grad(set_to_none=True)
                for indices in group.split(micro):
                    events = data['images'][indices].to(device)
                    height, width = events.shape[-2:]
                    anchors = anchors_for(height, width, self.settings['anchor_sizes'], device)
                    scale = torch.tensor([width, height, width, height], device=device)
                    targets = [data['boxes'][int(i)].to(device) / scale for i in indices]
                    loss = multibox_loss(model(events), anchors, targets)
                    if not torch.isfinite(loss):
                        raise ValueError('nonfinite BN-SNN training loss')
                    (loss * len(indices) / len(group)).backward()
                    total_loss += float(loss.detach()) * len(indices)
                    # State belongs to one event window, never to a later batch.
                    model.reset_state()
                torch.nn.utils.clip_grad_norm_(model.parameters(), recipe['grad_clip'], error_if_nonfinite=True)
                optimizer.step()
            print(f'BN-SNN epoch {epoch + 1}/{proxy["epochs"]}: loss={total_loss / len(order):.6f}', flush=True)

    @torch.no_grad()
    def predict(self, model, images, proxy, device):
        model.eval()
        results = []
        for events in images.split(self.settings['training']['microbatch_size']):
            events = events.to(device)
            height, width = events.shape[-2:]
            anchors = anchors_for(height, width, self.settings['anchor_sizes'], device)
            output = model(events)
            if not all(torch.isfinite(output[k]).all() for k in ('locations', 'confidences')):
                raise ValueError('nonfinite BN-SNN prediction')
            boxes = decode(output['locations'], anchors)
            scores = output['confidences'].softmax(-1)[..., 1]
            for box, score in zip(boxes, scores):
                valid = (score >= proxy['eval_conf']) & (box[:, 2:] > box[:, :2]).all(-1)
                box, score = box[valid], score[valid]
                indices = score.argsort(descending=True)[:self.settings['pre_nms_topk']]
                box, score = box[indices], score[indices]
                keep = nms(box, score, proxy['nms_iou'])[:self.settings['max_detections']]
                scale = torch.tensor([width, height, width, height], device=device)
                results.append(((box[keep] * scale).cpu(), score[keep].cpu()))
            model.reset_state()
        return results

    def postprocessing(self, proxy):
        return (f'SSD softmax drone; conf={proxy["eval_conf"]}; NMS IoU={proxy["nms_iou"]}; '
                f'topk={self.settings["pre_nms_topk"]}; max={self.settings["max_detections"]}')
