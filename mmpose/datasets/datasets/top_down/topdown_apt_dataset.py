from ...registry import DATASETS
from .topdown_coco_dataset import TopDownCocoDataset
import numpy as np

@DATASETS.register_module()
class TopDownAPTDataset(TopDownCocoDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        import poseConfig
        conf = poseConfig.conf

        self.use_gt_bbox = True
        flip_idx = list(range(conf.n_classes))
        pairs = []
        done = []
        for kk in conf.flipLandmarkMatches.keys():
            if int(kk) in done:
                continue
            pairs.append([int(kk),int(conf.flipLandmarkMatches[kk])])
            done.append(int(kk))
            done.append(int(conf.flipLandmarkMatches[kk]))
        self.ann_info['flip_pairs'] = pairs
        self.ann_info['joint_weights'] = np.ones([conf.n_classes])
        self.sigmas = np.ones([conf.n_classes])*0.6/10.0

    def _xywh2cs(self, x, y, w, h):
        """This encodes bbox(x,y,w,w) into (center, scale)

        Args:
            x, y, w, h

        Returns:
            tuple: A tuple containing center and scale.

            - center (np.ndarray[float32](2,)): center of the bbox (x, y).
            - scale (np.ndarray[float32](2,)): scale of the bbox w & h.
        """
        center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)
        scale = np.array([w / 200.0, h / 200.0], dtype=np.float32)
        return center, scale
