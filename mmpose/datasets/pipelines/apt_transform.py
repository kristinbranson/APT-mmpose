from mmpose.datasets.registry import PIPELINES
import numpy as np

@PIPELINES.register_module()
class APTtransform:
    """Data augmentation using APT's posetools.

    """


    def __init__(self,distort):
        import poseConfig
        self.conf = poseConfig.conf
        self.conf.normalize_img_mean = False
        self.conf.normalize_batch_mean = False
        self.distort = distort

    def __call__(self, results):
        import PoseTools as pt
        conf = self.conf
        if conf.is_multi:
            image, joints, mask = results['img'], results['joints'], results['mask']
            # assert mask[0].min(), 'APT transform only supports dummy masks'
            assert  not results['ann_info']['scale_aware_sigma'], 'APT doesnt support this'
            for jndx in range(len(joints)-1):
                # assert len(joints) ==2, "APT Transform is tested only for at most two scales"

                assert np.allclose(joints[jndx],joints[jndx+1],equal_nan=True), "APT transform is tested only for two identical scale inputs"
            jlen = len(joints)
            joints_in = joints[0][...,:2]
            occ_in = joints[0][...,2]
            joints_in[occ_in<1,:] = -100000
            image,joints_out,mask_out = pt.preprocess_ims(image[np.newaxis,...],joints_in[np.newaxis,...],conf,self.distort,conf.rescale,mask=mask[0][None,...])
            image = image.astype('float32')
            joints_out_occ = np.isnan(joints_out[0, ..., 0:1]) | (joints_out[0, ..., 0:1] < -1000)
            joints_out = np.concatenate([joints_out[0,...],(~joints_out_occ)*2],axis=-1)
            in_sz = results['ann_info']['image_size']
            out_sz = results['ann_info']['heatmap_size']
            assert all([round(in_sz/o)==in_sz/o for o in out_sz]), 'Output sizes should be integer multiples of input sizes'
            outs = [int(round(in_sz/o)) for o in out_sz]
            results['joints'] = [joints_out * osz / in_sz for osz in out_sz]
            results['mask'] = [mask_out[0,::o,::o]>0.5 for o in outs]

        else:
            image, joints, occ_in = results['img'], results['joints_3d'], results['joints_3d_visible']
            assert joints[:,2].max() < 0.00001, 'APT does not work 3d'
            occ_in = occ_in[:,0]
            joints_in = joints[:,:2]
            joints_in[occ_in<0.5,:] = -100000

            image,joints_out = pt.preprocess_ims(image[np.newaxis,...],joints_in[np.newaxis,...],conf,self.distort,conf.rescale)
            image = image.astype('float32')
            joints_out_occ = np.isnan(joints_out[0,...,0:1]) | (joints_out[0,...,0:1]<-1000)

            results['joints_3d'] = np.concatenate([joints_out[0,...],np.zeros_like(joints_out[0,:,:1])],1)
            results['joints_3d_visible'] = np.concatenate([1-joints_out_occ,1-joints_out_occ,np.zeros_like(joints_out_occ)],1)

        results['img'] = np.clip(image[0,...],0,255).astype('uint8')
        return results

