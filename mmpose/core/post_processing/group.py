# ------------------------------------------------------------------------------
# Adapted from https://github.com/princeton-vl/pose-ae-train/
# Original licence: Copyright (c) 2017, umich-vl, under BSD 3-Clause License.
# ------------------------------------------------------------------------------

import numpy as np
import torch
from munkres import Munkres
import multiprocessing
from mmpose.core.evaluation import post_dark_udp


def _py_max_match(scores):
    """Apply munkres algorithm to get the best match.

    Args:
        scores(np.ndarray): cost matrix.

    Returns:
        np.ndarray: best match.
    """
    m = Munkres()
    tmp = m.compute(scores)
    tmp = np.array(tmp).astype(int)
    return tmp


def _match_by_tag(inp, params):
    """Match joints by tags. Use Munkres algorithm to calculate the best match
    for keypoints grouping.

    Note:
        number of keypoints: K
        max number of people in an image: M (M=30 by default)
        dim of tags: L
            If use flip testing, L=2; else L=1.

    Args:
        inp(tuple):
            tag_k (np.ndarray[KxMxL]): tag corresponding to the
                top k values of feature map per keypoint.
            loc_k (np.ndarray[KxMx2]): top k locations of the
                feature maps for keypoint.
            val_k (np.ndarray[KxM]): top k value of the
                feature maps per keypoint.
        params(Params): class Params().

    Returns:
        np.ndarray: result of pose groups.
    """
    assert isinstance(params, _Params), 'params should be class _Params()'

    tag_k, loc_k, val_k = inp

    default_ = np.zeros((params.num_joints, 3 + tag_k.shape[2]),
                        dtype=np.float32)

    joint_dict = {}
    tag_dict = {}
    for i in range(params.num_joints):
        idx = params.joint_order[i]

        tags = tag_k[idx]
        joints = np.concatenate((loc_k[idx], val_k[idx, :, None], tags), 1)
        mask = joints[:, 2] > params.detection_threshold

        # APT update to have minimum number of detections
        if np.count_nonzero(mask) < params.min_num_people:
            order = np.flip(np.argsort(joints[:, 2]))
            mask[order[:params.min_num_people]] = True

        tags = tags[mask]
        joints = joints[mask]

        if joints.shape[0] == 0:
            continue

        if i == 0 or len(joint_dict) == 0:
            for tag, joint in zip(tags, joints):
                key = tag[0]
                joint_dict.setdefault(key, np.copy(default_))[idx] = joint
                tag_dict[key] = [tag]
        else:
            grouped_keys = list(joint_dict.keys())[:params.max_num_people]
            grouped_tags = [np.mean(tag_dict[j], axis=0) for j in grouped_keys]
            prev_joints = params.joint_order[:i]
            grouped_joints = [joint_dict[j][prev_joints,:2] for j in grouped_keys]
            grouped_joints = [np.mean(g[g[:,:].sum(axis=1)>0,:],axis=0) for g in grouped_joints]

            if (params.ignore_too_much
                    and len(grouped_keys) == params.max_num_people):
                continue

            diff = joints[:, None, 3:] - np.array(grouped_tags)[None, :, :]
            diff_normed = np.linalg.norm(diff, ord=2, axis=2)

            if params.dist_grouping:
                # For APT, add distance based condition to avoid joining far off points. AE is able to give different tag values to close by animals, but can at times give same tag value to animals that are far apart

                # Find the distance of current joints to previously grouped joints.
                diff_joint = joints[:,None,:2] - np.array(grouped_joints)[None,:,:]
                diff_joint_normed = np.linalg.norm(diff_joint,ord=2,axis=2)
                # Take the 25th percentile distance as the cutoff distance so that this factor can scale to different sized animals/resolutions.
                med_joint = np.percentile(diff_joint_normed.flatten(), 25)
                dd = diff_joint_normed-med_joint
                # The distance factor joint_logistic will be close to 0 if the dd<<med_joint, while will be close to 4 if dd>>med_joint.
                joint_logistic = 4 / (1 + np.exp(-dd / med_joint*4))
                assert ~np.any(np.isnan(joint_logistic)), 'This should not happen'

                diff_normed = diff_normed + joint_logistic

            diff_saved = np.copy(diff_normed)

            if params.use_detection_val:
                diff_normed = np.round(diff_normed) * 100 - joints[:, 2:3]

            num_added = diff.shape[0]
            num_grouped = diff.shape[1]

            if num_added > num_grouped:
                diff_normed = np.concatenate(
                    (diff_normed,
                     np.zeros((num_added, num_added - num_grouped),
                              dtype=np.float32) + 1e10),
                    axis=1)

            pairs = _py_max_match(diff_normed)
            for row, col in pairs:
                if (row < num_added and col < num_grouped
                        and diff_saved[row][col] < params.tag_threshold):
                    key = grouped_keys[col]
                    joint_dict[key][idx] = joints[row]
                    tag_dict[key].append(tags[row])
                else:
                    key = tags[row][0]
                    joint_dict.setdefault(key, np.copy(default_))[idx] = \
                        joints[row]
                    tag_dict[key] = [tags[row]]

    ans = np.array([joint_dict[i] for i in joint_dict]).astype(np.float32)
    return ans


class _Params:
    """A class of parameter.

    Args:
        cfg(Config): config.
    """

    def __init__(self, cfg):
        self.num_joints = cfg['num_joints']
        self.max_num_people = cfg['max_num_people']

        self.detection_threshold = cfg['detection_threshold']
        self.tag_threshold = cfg['tag_threshold']
        self.use_detection_val = cfg['use_detection_val']
        self.ignore_too_much = cfg['ignore_too_much']

        if 'min_num_people' in cfg.keys():
            self.min_num_people = cfg['min_num_people']
        else:
            self.min_num_people = 0

        if 'dist_grouping' in cfg.keys():
            self.dist_grouping = cfg['dist_grouping']
        else:
            self.dist_grouping = False

        if self.num_joints == 17:
            self.joint_order = [
                i - 1 for i in
                [1, 2, 3, 4, 5, 6, 7, 12, 13, 8, 9, 10, 11, 14, 15, 16, 17]
            ]
        else:
            self.joint_order = list(np.arange(self.num_joints))


class HeatmapParser:
    """The heatmap parser for post processing."""

    def __init__(self, cfg):
        self.params = _Params(cfg)
        self.tag_per_joint = cfg['tag_per_joint']
        self.pool = torch.nn.MaxPool2d(cfg['nms_kernel'], 1,
                                       cfg['nms_padding'])
        self.use_udp = cfg.get('use_udp', False)
        self.dist_grouping = cfg.get('dist_grouping',False)

    def nms(self, heatmaps):
        """Non-Maximum Suppression for heatmaps.

        Args:
            heatmap(torch.Tensor): Heatmaps before nms.

        Returns:
            torch.Tensor: Heatmaps after nms.
        """

        maxm = self.pool(heatmaps)
        maxm = torch.eq(maxm, heatmaps).float()
        heatmaps = heatmaps * maxm

        return heatmaps

    def match(self, tag_k, loc_k, val_k):
        """Group keypoints to human poses in a batch.

        Args:
            tag_k (np.ndarray[NxKxMxL]): tag corresponding to the
                top k values of feature map per keypoint.
            loc_k (np.ndarray[NxKxMx2]): top k locations of the
                feature maps for keypoint.
            val_k (np.ndarray[NxKxM]): top k value of the
                feature maps per keypoint.

        Returns:
            list
        """

        def _match(x):
            return _match_by_tag(x, self.params)

        return list(map(_match, zip(tag_k, loc_k, val_k)))

    def top_k(self, heatmaps, tags):
        """Find top_k values in an image.

        Note:
            batch size: N
            number of keypoints: K
            heatmap height: H
            heatmap width: W
            max number of people: M
            dim of tags: L
                If use flip testing, L=2; else L=1.

        Args:
            heatmaps (torch.Tensor[NxKxHxW])
            tags (torch.Tensor[NxKxHxWxL])

        Return:
            dict: A dict containing top_k values.

            - tag_k (np.ndarray[NxKxMxL]):
                tag corresponding to the top k values of
                feature map per keypoint.
            - loc_k (np.ndarray[NxKxMx2]):
                top k location of feature map per keypoint.
            - val_k (np.ndarray[NxKxM]):
                top k value of feature map per keypoint.
        """
        heatmaps = self.nms(heatmaps)
        N, K, H, W = heatmaps.size()
        heatmaps = heatmaps.view(N, K, -1)
        val_k, ind = heatmaps.topk(self.params.max_num_people, dim=2)

        tags = tags.view(tags.size(0), tags.size(1), W * H, -1)
        if not self.tag_per_joint:
            tags = tags.expand(-1, self.params.num_joints, -1, -1)

        tag_k = torch.stack(
            [torch.gather(tags[..., i], 2, ind) for i in range(tags.size(3))],
            dim=3)

        x = ind % W
        y = ind // W

        ind_k = torch.stack((x, y), dim=3)

        ans = {
            'tag_k': tag_k.cpu().numpy(),
            'loc_k': ind_k.cpu().numpy(),
            'val_k': val_k.cpu().numpy()
        }

        return ans

    @staticmethod
    def adjust(ans, heatmaps):
        """Adjust the coordinates for better accuracy.

        Note:
            batch size: N
            number of keypoints: K
            heatmap height: H
            heatmap width: W

        Args:
            ans (list(np.ndarray)): Keypoint predictions.
            heatmaps (torch.Tensor[NxKxHxW]): Heatmaps.
        """
        _, _, H, W = heatmaps.shape
        for batch_id, people in enumerate(ans):
            for people_id, people_i in enumerate(people):
                for joint_id, joint in enumerate(people_i):
                    if joint[2] > 0:
                        x, y = joint[0:2]
                        xx, yy = int(x), int(y)
                        tmp = heatmaps[batch_id][joint_id]
                        if tmp[min(H - 1, yy + 1), xx] > tmp[max(0, yy - 1),
                                                             xx]:
                            y += 0.25
                        else:
                            y -= 0.25

                        if tmp[yy, min(W - 1, xx + 1)] > tmp[yy,
                                                             max(0, xx - 1)]:
                            x += 0.25
                        else:
                            x -= 0.25
                        ans[batch_id][people_id, joint_id,
                                      0:2] = (x + 0.5, y + 0.5)
        return ans

    @staticmethod
    def refine(heatmap, tag, keypoints, use_udp=False, adjust_dist=False):
        """Given initial keypoint predictions, we identify missing joints.

        Note:
            number of keypoints: K
            heatmap height: H
            heatmap width: W
            dim of tags: L
                If use flip testing, L=2; else L=1.

        Args:
            heatmap: np.ndarray(K, H, W).
            tag: np.ndarray(K, H, W) |  np.ndarray(K, H, W, L)
            keypoints: np.ndarray of size (K, 3 + L)
                        last dim is (x, y, score, tag).
            use_udp: bool-unbiased data processing

        Returns:
            np.ndarray: The refined keypoints.
        """

        K, H, W = heatmap.shape
        if len(tag.shape) == 3:
            tag = tag[..., None]

        tags = []
        for i in range(K):
            if keypoints[i, 2] > 0:
                # save tag value of detected keypoint
                x, y = keypoints[i][:2].astype(int)
                x = np.clip(x, 0, W - 1)
                y = np.clip(y, 0, H - 1)
                tags.append(tag[i, y, x])

        # mean tag of current detected people
        prev_tag = np.mean(tags, axis=0)
        ans = []

        # For APT, add distances
        if adjust_dist:
            # Add a distance penalty during refining. The penalty is close to 0 at pixels close to the centroid of the group. The max value the penalty can take is 4
            med_tag = keypoints[:,:2]
            med_tag = np.mean(med_tag[med_tag[:,0]>0.001,:],axis=0)
            x_dist, y_dist = np.meshgrid(range(W),range(H))
            dist_mat = np.sqrt((x_dist-med_tag[0])**2 + (y_dist-med_tag[1])**2)
            dist_mat = 8*(1/(1+np.exp(-dist_mat/(H+W)*8))-0.5)
        else:
            dist_mat = np.zeros([H,W])

        for _heatmap, _tag in zip(heatmap, tag):
            # distance of all tag values with mean tag of
            # current detected people
            distance_tag = (((_tag -
                              prev_tag[None, None, :])**2).sum(axis=2)**0.5)
            # norm_heatmap = _heatmap - np.round(distance_tag) - dist_mat
            # rounding off as int is faster. MK 20210122
            norm_heatmap = _heatmap - (distance_tag+0.5).astype('int') - dist_mat

            # find maximum position
            y, x = np.unravel_index(np.argmax(norm_heatmap), _heatmap.shape)
            xx = x.copy()
            yy = y.copy()
            # detection score at maximum position
            val = _heatmap[y, x]
            if not use_udp:
                # offset by 0.5
                x += 0.5
                y += 0.5

            # add a quarter offset
            if _heatmap[yy, min(W - 1, xx + 1)] > _heatmap[yy, max(0, xx - 1)]:
                x += 0.25
            else:
                x -= 0.25

            if _heatmap[min(H - 1, yy + 1), xx] > _heatmap[max(0, yy - 1), xx]:
                y += 0.25
            else:
                y -= 0.25

            ans.append((x, y, val))
        ans = np.array(ans)

        if ans is not None:
            for i in range(K):
                # add keypoint if it is not detected
                if ans[i, 2] > 0 and keypoints[i, 2] == 0:
                    keypoints[i, :3] = ans[i, :3]

        return keypoints

    def parse(self, heatmaps, tags, adjust=True, refine=True):
        """Group keypoints into poses given heatmap and tag.

        Note:
            batch size: N
            number of keypoints: K
            heatmap height: H
            heatmap width: W
            dim of tags: L
                If use flip testing, L=2; else L=1.

        Args:
            heatmaps (torch.Tensor[NxKxHxW]): model output heatmaps.
            tags (torch.Tensor[NxKxHxWxL]): model output tagmaps.

        Returns:
            tuple: A tuple containing keypoint grouping results.

            - ans (list(np.ndarray)): Pose results.
            - scores (list): Score of people.
        """
        ans = self.match(**self.top_k(heatmaps, tags))

        if adjust:
            if self.use_udp:
                for i in range(len(ans)):
                    if ans[i].shape[0] > 0:
                        ans[i][..., :2] = post_dark_udp(
                            ans[i][..., :2].copy(), heatmaps[i:i + 1, :])
            else:
                ans = self.adjust(ans, heatmaps)

        scores = [i[:, 2].mean() for i in ans[0]]

        if refine:
            ans = ans[0]
            # for every detected person
            heatmap_numpy = heatmaps[0].cpu().numpy()
            tag_numpy = tags[0].cpu().numpy()
            if not self.tag_per_joint:
                tag_numpy = np.tile(tag_numpy,
                                    (self.params.num_joints, 1, 1, 1))
            for i in range(len(ans)):
                ans[i] = self.refine(
                heatmap_numpy, tag_numpy, ans[i], use_udp=self.use_udp, adjust_dist=self.dist_grouping)
            ans = [ans]

        return ans, scores
