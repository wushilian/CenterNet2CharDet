import numpy as np

def cal_iou(bbox1,bbox2):
    '''
    
    :param bbox1: [N,4]
    :param bbox2: [4]
    :return:
    '''
    assert bbox1.shape[1]==4 and bbox2.shape[0]==4
    assert len(bbox1.shape)==2 and len(bbox2.shape)==1
    bbox2=np.tile(bbox2[np.newaxis,:],[bbox1.shape[0],1])
    
    in_w=np.minimum(bbox1[:,2],bbox2[:,2])-np.maximum(bbox1[:,0],bbox2[:,0])
    in_h = np.minimum(bbox1[:, 3], bbox2[:, 3]) - np.maximum(bbox1[:, 1], bbox2[:, 1])
    inter=in_h*in_w
    
    area1=(bbox1[:,2]-bbox1[:,0])*(bbox1[:,3]-bbox1[:,1])
    area2 = (bbox2[:, 2] - bbox2[:, 0]) * (bbox2[:, 3] - bbox2[:, 1])
    union=area1+area2-inter
    iou=inter/union
    return iou

class DetEval:
    def __init__(self,overlap_thresh=0.7):
        self.TP=0
        self.FP=0
        self.FN=0
        self.GT=0
        self.ALL=0
        self.overlap_thresh=overlap_thresh
    def __call__(self,pred_bbox,gt_bbox ):
        assert len(pred_bbox.shape)==2 and len(gt_bbox.shape)==2
        self.GT+=gt_bbox.shape[0]
        self.ALL+=pred_bbox.shape[0]
        

        fn_bbox=[]
        for i in range(gt_bbox.shape[0]):
            iou=cal_iou(pred_bbox,gt_bbox[i])
            thresh_iou=iou[iou>self.overlap_thresh]
            if len(thresh_iou)==0:
                fn_bbox.append(gt_bbox[i])
            else:
                self.TP+=1
                self.FP+=len(iou)-1
            zero_iou=iou[iou>0.1]
            if len(zero_iou)==0:
                self.FN+=1
        return np.array(fn_bbox)
    
    def reset(self):
        self.GT=0
        self.FN=0
        self.TP=0
        self.FP=0
        self.ALL=0
        
    def eval(self):
        acc=self.TP/(self.ALL+1e-10)
        recall=self.TP/(self.GT+1e-10)
        mis_rate=self.FN/(self.GT+1e-10)
        self.res={'acc':acc,'recall':recall,'mis_rate':mis_rate}
        print('--------------------------准确率:%.4f(%d/%d),检出率:%.4f(%d/%d),漏检率:%.4f(%d/%d)-------------------------'
              %(acc,self.TP,self.ALL,recall,self.TP,self.GT,mis_rate,self.FN,self.GT))
        self.reset()
        
    
if __name__=='__main__':
    bbox1=np.array([[0,0,10,10],[0,0,20,20]])
    bbox2=np.array([0,0,5,5])
    print(cal_iou(bbox1,bbox2))