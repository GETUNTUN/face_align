'''
侧脸识别
'''
def face_correction(shape):
    '''
    :param shape: 人脸关键点
    :return: 1表示正脸，0表示侧脸
    '''
    #左右脸宽度比
    x_l = shape.part(27).x - shape.part(0).x
    x_r = shape.part(16).x - shape.part(27).x
    x_ratio = x_l/x_r

    #左右内眼角宽度比
    f_l = shape.part(27).x - shape.part(39).x
    f_r = shape.part(42).x - shape.part(27).x
    f_ratio = f_l/f_r

    if x_ratio >= 0.5 and x_ratio <= 2 and f_ratio >= 0.5 and f_ratio <= 2:
        true_face = 1
    else:
        true_face = 0
    return true_face

