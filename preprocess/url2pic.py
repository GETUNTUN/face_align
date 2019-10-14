import urllib.request as ul
import numpy as np
import cv2
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
def url2pic(url):
    rsp = ul.urlopen(url)
    image = np.asarray(bytearray(rsp.read()), dtype='uint8')
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

# url = 'https://ss0.baidu.com/6ONWsjip0QIZ8tyhnq/it/u=3565326027,4113253682&fm=173&app=25&f=JPEG?w=640&h=341&s=07A06CA64E333196906CD8B50300F0C1'
# img = url2pic(url)
# print(img)