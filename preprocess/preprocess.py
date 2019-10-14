# coding=utf-8

from __future__ import absolute_import, print_function

import os
import dlib
import warnings

from align_face import deal_img
from crop import face_correction

warnings.filterwarnings("ignore")


def init_landmark():
    '''
    :return: detector:检测图中所有人脸模型
            predictor:检测人脸关键点模型
    '''
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'shape_predictor_68_face_landmarks.dat')
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(model_path)
    return [detector, predictor]


def preprocess(url, landmark_model):
    '''
    :param url: 图片路径
    :param landmark_model: [detector,prediction]
    :return: 异常人脸返回人脸数量，一张正常人脸返回[图片数组,68个关键点]
    '''
    img = deal_img(url, landmark_model)
    if isinstance(img,int):
        return img
    else:
        dets = landmark_model[0](img, 1)
        if len(dets) != 1:
            return len(dets)
        else:
            face = dets[0]
            shape = landmark_model[1](img, face)
            if face_correction(shape):
                return img, shape
            else:
                return 0

def main():
    url_list = ['https://pic.igengmei.com/2019/03/18/1117/217145886f92-w']
    landmark_model = init_landmark()
    for url in url_list:
        imgs = preprocess(url, landmark_model)
        print(imgs)

if __name__ == '__main__':
    main()






