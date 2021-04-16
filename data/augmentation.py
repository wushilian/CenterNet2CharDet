import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.polys import Polygon, PolygonsOnImage
import cv2
from utils.read_data import read_labelme

class AugPoly:
    def __init__(self):
        st = lambda aug: iaa.Sometimes(0.5, aug)
        self.seq = iaa.Sequential([
            st(iaa.Pad(percent=((0, 0.2), (0, 0.2), (0, 0.2), (0, 0.2)), keep_size=False)),
            #
            #st(iaa.Crop(percent=([0.0, 0.1], [0.00, 0.1], [0.0, 0.1], [0.0, 0.1]), keep_size=False)),
            st(iaa.Affine(scale=(0.9, 1.0), rotate=(-30, 30), shear=(-5, 5),
                          translate_px={"x": (-30, 30), "y": (-10, 10)},
                          fit_output=True)),
            # st(iaa.PerspectiveTransform((0,0.1),fit_output=True)),
            # st(iaa.MultiplyAndAddToBrightness(mul=(0.6, 1.5), add=(0, 30))),
            st(iaa.ChangeColorTemperature(kelvin=(3000, 9100))),
            st(iaa.LinearContrast((0.75, 1.5))),
            st(iaa.GaussianBlur((0, 0.2))),
            # st(iaa.PerspectiveTransform(scale=0.05,)),
            st(iaa.AddToHueAndSaturation((-20, 20))),
            #
            st(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 16),
                                         per_channel=True)),  # add gaussian noise to images
            # # # #st(iaa.Dropout((0.0, 0.1), per_channel=0.5)),  # randomly remove up to 10% of the pixels
            # # # change brightness of images (by -10 to 10 of original value)
            st(iaa.Add((-40, 40), per_channel=True)),
            # # change brightness of images (50-150% of original value)
            st(iaa.Multiply((0.5, 1.5), per_channel=True)),
        ])
    def aug(self,im,bboxes,viz=True):
        assert len(bboxes.shape)==3
        assert bboxes.shape[1]==4

        poly_on_img = PolygonsOnImage([Polygon(bbox) for bbox in bboxes], shape=im.shape)
        if viz:
            res = poly_on_img.draw_on_image(im)
            cv2.imshow('ori', res)
            cv2.waitKey(0)
        imgs_aug, poly_on_img_aug = self.seq(image=im, polygons=poly_on_img,)
        res_bboxes=poly_on_img_aug.to_xy_array()
        if bboxes.shape[0]*bboxes.shape[1]!=res_bboxes.shape[0]:
            print('aug error','before:',bboxes.shape,'  after:',res_bboxes.shape)
            return im,bboxes
        #print('before:',bboxes.shape,'  after:',res_bboxes.shape)
        res_bboxes=np.reshape(res_bboxes,bboxes.shape)
        if viz:
            res = poly_on_img_aug.draw_on_image(imgs_aug)
            cv2.imshow('a', res)
            cv2.waitKey(0)
        return imgs_aug,res_bboxes
    
def test_aug():
    im=cv2.imread('/media/wsl/SB@data/dataset/瓶盖分类/dataset/单字检测/聪明盖/0_0.jpg')
    bboxes=read_labelme('/media/wsl/SB@data/dataset/瓶盖分类/dataset/单字检测/聪明盖/0_0.json')
    poly_on_img=PolygonsOnImage([Polygon(bbox) for bbox in bboxes],shape=im.shape)
    
    st = lambda aug: iaa.Sometimes(1, aug)
    seq = iaa.Sequential([
        #st(iaa.Pad(percent=((0, 0.2), (0, 0.2), (0, 0.2), (0, 0.2)), keep_size=False)),
        #
        # # st(iaa.Crop(percent=([0.0, 0.3], [0.00, 0.1], [0.0, 0.3], [0.0, 0.1]), keep_size=False)),
        st(iaa.Affine(scale=(0.9, 1.0), rotate=(-45, 45), shear=(-5, 5), translate_px={"x": (-16, 16), "y": (-10, 10)},
                      fit_output=True)),
        st(iaa.Add(value=(-10, 10), per_channel=True)),
        # st(iaa.PerspectiveTransform((0,0.1),fit_output=True)),
        # st(iaa.MultiplyAndAddToBrightness(mul=(0.6, 1.5), add=(0, 30))),
        st(iaa.ChangeColorTemperature(kelvin=(3000, 9100))),
        st(iaa.Sharpen(0, 0.1)),
        st(iaa.GaussianBlur((0, 1))),
        st(iaa.AddToHueAndSaturation((-2, 2))),
        st(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.2), per_channel=True)),  # add gaussian noise to images
    ])
    for i in range(10):
        imgs_aug,poly_on_img_aug=seq(image=im,polygons=poly_on_img)
        
        res=poly_on_img_aug.draw_on_image(imgs_aug)
        a=poly_on_img_aug.to_xy_array()
        print(a.shape)
        cv2.imshow('a',res)
        cv2.waitKey(0)
        
if __name__=='__main__':
    test_aug()