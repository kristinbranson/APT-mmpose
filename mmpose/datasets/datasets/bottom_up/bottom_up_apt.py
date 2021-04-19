from mmpose.datasets.builder import DATASETS
from .bottom_up_base_dataset import BottomUpBaseDataset
from .bottom_up_coco import BottomUpCocoDataset
import numpy as np
import xtcocotools

@DATASETS.register_module()
class BottomUpAPTDataset(BottomUpCocoDataset):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        import poseConfig
        conf = poseConfig.conf
        flip_idx = list(range(conf.n_classes))
        for kk in conf.flipLandmarkMatches.keys():
            flip_idx[int(kk)] = conf.flipLandmarkMatches[kk]
        self.ann_info['flip_index'] = flip_idx
        self.ann_info['joint_weights'] = np.ones([conf.n_classes])
        self.sigmas = np.ones([conf.n_classes])*0.6/10.0
        self.ann_info['joint_weights'] = np.ones([self.ann_info['num_joints'],1])
        self.conf = conf


    def _get_mask(self, anno, idx):
        # Masks are created during image generation.
        conf = self.conf
        coco = self.coco
        img_info = coco.loadImgs(self.img_ids[idx])[0]
        m = np.zeros((img_info['height'], img_info['width']), dtype=np.float32)
        if not conf.multi_loss_mask:
            return m<0.5

        for obj in anno:
            if 'segmentation' in obj:
                if obj['iscrowd']:
                    rle = xtcocotools.mask.frPyObjects(obj['segmentation'],
                                                       img_info['height'],
                                                       img_info['width'])
                    m += xtcocotools.mask.decode(rle)
                else:
                    rles = xtcocotools.mask.frPyObjects(
                        obj['segmentation'], img_info['height'],
                        img_info['width'])
                    for rle in rles:
                        m += xtcocotools.mask.decode(rle)

        return m > 0.5




