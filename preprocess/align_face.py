import numpy as np
import scipy.ndimage

import PIL.Image
from url2pic import url2pic
from urllib.request import urlopen
from io import BytesIO

def image_align(src_file, face_landmarks, output_size = 512, transform_size=512, enable_padding=True, x_scale=1, y_scale=1, em_scale=0.1, alpha=False):

    lm = np.array(face_landmarks)
    lm_eye_left = lm[36:42]
    lm_eye_right = lm[42:48]
    lm_mouth_outer = lm[48:60]

# calculate auxiliary vectors

    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right)*0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right)*0.5
    eye_to_mouth = mouth_avg - eye_avg

# choose oriented crop rectangle

    x = eye_to_eye - np.flipud(eye_to_mouth)*[-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) *2, np.hypot(*eye_to_mouth)*1.8)
    x *= x_scale
    y = np.flipud(x)*[-y_scale, y_scale]
    c = eye_avg + eye_to_mouth*em_scale
    quad = np.stack([c - x - y, c-x+y, c+x+y, c+x-y])
    qsize = np.hypot(*x)*2

# load in-the-wild image

    response = urlopen(src_file)
    img = PIL.Image.open(BytesIO(response.read()))

# shrink

    shrink = int(np.floor(qsize/output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

# crop

    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
            min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

# pad

    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
           int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0),
           max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                          1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = np.uint8(np.clip(np.rint(img), 0, 255))
        if alpha:
            mask = 1 - np.clip(3.0 * mask, 0.0, 1.0)
            mask = np.uint8(np.clip(np.rint(mask * 255), 0, 255))
            img = np.concatenate((img, mask), axis=2)
            img = PIL.Image.fromarray(img, 'RGBA')
        else:
            img = PIL.Image.fromarray(img, 'RGB')
        quad += pad[:2]

# transform

    img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)
    return img

class LandmarksDetector:
    def __init__(self, landmask_model):
        """
        :param predictor_model_path: path to shape_predictor_68_face_landmarks.dat file
        """
        self.detector = landmask_model[0]#dlib.get_frontal_face_detector() # cnn_face_detection_model_v1 also can be used
        self.shape_predictor = landmask_model[1]#(predictor_model_path)

    def get_landmarks(self, url):
        img = url2pic(url)
        dets = self.detector(img, 1)
        landmarks_list = []
        if len(dets)!=1:
            return len(dets)
        else:
            for detection in dets:
                try:
                    face_landmarks = [(item.x, item.y) for item in self.shape_predictor(img, detection).parts()]
                    # yield face_landmarks
                    landmarks_list.append(face_landmarks)
                    return landmarks_list
                except:
                    print("Exception in get_landmarks()!")



def deal_img(raw_img_path,landmark_model):
    landmarks_detector = LandmarksDetector(landmark_model)
    dets = landmarks_detector.get_landmarks(raw_img_path)
    if type(dets)==np.int:
        return dets
    else:
        for face_landmarks in dets:
            try:
                img = image_align(raw_img_path, face_landmarks, output_size=512, x_scale=1,
                    y_scale=1, em_scale=0.1, alpha=False)
                img = np.array(img)
            except:
                print("Exception in face alignment!")
        return img
if __name__ == '__main__':
    raw_dir = '/home/pc/桌面/result_img.png'
    # raw_dir ='/home/pc/workspace/fashionai/oldrecognition/data/meitu_0829_71826/meitu_0829/img/http:__tuchong.pstatp.com_13478_l_162532.jpg'
    img = deal_img(raw_dir)
    img.show()
