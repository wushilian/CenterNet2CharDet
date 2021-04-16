import json
import numpy as np
import cv2

def read_labelme(json_file):
    data = open(json_file, "r")
    data = json.load(data)
    shapes = data["shapes"]
    bbox = []
    for shape in shapes:
        pt = np.array(shape['points'], dtype=np.int)
        if len(pt)==2:
            if pt[0][0]>pt[1][0]:
                pt=np.array([pt[1],pt[0]])
            pt=np.array([pt[0],[pt[1][0],pt[0][1]],pt[1],[pt[0][0],pt[1][1]]])
            # rect = cv2.minAreaRect(pt)  # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
            # pt = cv2.boxPoints(rect)
        else:
            rect = cv2.minAreaRect(pt)  # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
            pt = cv2.boxPoints(rect)
        # pt=sort_pts(pt)
        assert len(pt) == 4
        bbox.append(pt)
    bbox=np.array(bbox)
    bbox=bbox.reshape(-1,4,2)
    return bbox

if __name__=='__main__':
    import cv2
    im=cv2.imread('/media/wsl/SB@data/dataset/hoke/seg/error/314.jpg')
    bb=read_labelme('/media/wsl/SB@data/dataset/hoke/seg/error/314.json')
    viz_img = cv2.polylines(im, bb.astype(np.int), isClosed=True, color=(255, 0, 0), thickness=1)
    cv2.imshow('a', viz_img)
    cv2.waitKey(0)
    print(bb)