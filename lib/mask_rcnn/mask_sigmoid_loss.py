# --------------------------------------------------------
# Mask R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Tiansheng Gan
# --------------------------------------------------------
import caffe
import numpy as np

DEBUG = False

class MaskSigmoidLossLayer(caffe.Layer):
    """
    Sigmoid loss for mask-rcnn
    """

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 4:
            raise Exception("Need three inputs to compute distance.")
        #self.loss = 0

    def reshape(self, bottom, top):
        # check input dimensions match
        #if bottom[0].count != bottom[2].count:
        #    raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs

        # loss output is scalar
        top[0].reshape(1)
        self.diff = None
        self.ind = None
        self.delta = 1e-5

    def get(self, x):
        ind0 = np.where(x >= 0)
        ind1 = np.where(x < 0)
        ret = np.zeros_like(x)
        # test_sigmoid = 1. / ( 1 + np.exp(-mask_pred))
        ret[ind0] = 1. / (1. + np.exp(-x[ind0]))
        ret[ind1] = np.exp(x[ind1]) / (1. + np.exp(x[ind1]))
        # print gt_mask.dtype
        #ret = 1. / ( 1 + np.exp(-x))
        assert np.all(ret > 0.)
        return ret

    def gradient_check(self, x, y):
        x1 = x + self.delta
        x2 = x - self.delta
        x1 = self.get(x1)
        x2 = self.get(x2)

        N, W, H = y.shape

        loss1 = y * np.log(x1) + (1. - y) * np.log(1. - x1)
        loss2 = y * np.log(x2) + (1. - y) * np.log(1. - x2)

        loss = (-loss1 + loss2) / 2. / self.delta
        #loss = -np.sum(loss) / N / W / H
        return loss

    def forward(self, bottom, top):
        #rois(0, x1, y1, x2, y2)
        #rois = bottom[4].data

        #gt bbox
        gt_rois = bottom[3].data

        #label
        labels = bottom[1].data

        #gt_mask
        gt_mask = bottom[2].data

        #data
        data = bottom[0].data

        ind = np.where(labels>0)
        labels = labels[ind].astype(int) - 1
        self.ind = zip(ind[0], labels)

        num_box = len(ind)
        mask_pred = data[ind[0], labels, :, :]
        gt_mask = gt_mask[ind]

        assert mask_pred.shape == gt_mask.shape
        assert len(labels) == mask_pred.shape[0]

        if DEBUG:
            print '================ MaskSigmoidLossLayer ==================='
            print 'label:{},{}'.format(len(labels), labels)
            print 'data shape:{}'.format(data.shape)
            print 'mask_pred shape:{}'.format(mask_pred.shape)
            print 'gt_mask shape:{}'.format(gt_mask.shape)
            #print 'rois shape:{}'.format(rois.shape)
            print 'gt_rois shape:{}'.format(gt_rois.shape)
            print 'len(ind):{} len(labels):{}'.format(len(ind), len(labels))
            #print 'ind:{}'.format(self.ind)

        #gc_diff = self.gradient_check(mask_pred, gt_mask)

        ind0 = np.where(mask_pred>=0)
        ind1 = np.where(mask_pred<0)
        #mask_pred_ = np.zeros_like(mask_pred)
        #mask_pred = 1. / ( 1 + np.exp(-mask_pred))
        mask_pred[ind0] = 1. / (1. + np.exp(-mask_pred[ind0]))
        mask_pred[ind1] = np.exp(mask_pred[ind1]) / (1. + np.exp(mask_pred[ind1]))
        #print gt_mask.dtype
        assert np.all(mask_pred > 0.)

        #mask_pred[mask_pred>=0.5] = 1
        #mask_pred[mask_pred<0.5] = 0
        #ind0 = np.where(gt_mask > 0)
        #ind1 = np.where(gt_mask == 0)

        N, W, H = gt_mask.shape
        self.diff = (mask_pred - gt_mask) / N / W / H
        #print 'diff shape:{}'.format(self.diff.shape)
        #print 'gc.diff:{}'.format(np.sum(np.abs(mask_pred - gt_mask - gc_diff)))
        #print 'abs.diff:{}, {}'.format(np.sum(np.abs(gc_diff)) , np.sum(np.abs(mask_pred - gt_mask)))
        #print 'rel_diff:{}'.format(np.sum(np.abs(mask_pred - gt_mask - gc_diff)/np.maximum(np.abs(gc_diff) , np.abs(mask_pred - gt_mask)) )/ N / W / H)

        loss = gt_mask * np.log(np.maximum(mask_pred, self.delta)) + (1. - gt_mask) * np.log(np.maximum(1. - mask_pred, self.delta))
        loss = -np.sum(loss) / N / W / H
        assert loss > 0

        if DEBUG:
            print 'loss.shape:{}'.format(loss.shape)
        top[0].data[...] = loss


        print 'loss:{}, total:{}'.format(loss, N*W*H)

    def backward(self, top, propagate_down, bottom):
        if propagate_down[0]:
            #print 'diff shape:{}'.format(bottom[0].diff.shape)
            #diff_before = bottom[0].diff[...]
            idx = 0
            for x, y in self.ind:
                #print 'diff axis:{},{},{}'.format(x, y, self.diff)
                bottom[0].diff[x, y, :, :] = self.diff[idx]
                idx += 1
            #print 'diff shape:{}'.format(bottom[0].diff[bottom[0].diff!=0].shape)
            #print 'diff before:{}'.format(diff_before)
            #print 'diff difference:{}'.format(bottom[0].diff[diff_before !=bottom[0].diff])