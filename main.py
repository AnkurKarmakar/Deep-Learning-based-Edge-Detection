import cv2 as cv
import numpy as np
from utils.preprocessing import CannyP
from utils.preprocessing import CropLayer
import sys
if __name__ == "__main__":
    # get image path
    if(len(sys.argv) > 1):
        src_path = sys.argv[1]
    else:
        src_path = "testdata/ankur.jpg"
    # read image
    img = cv.imread(src_path, 1) 
    if(img is None):
        print("Image not read properly")
        sys.exit(0)
    # initialize preprocessing object
    obj = CannyP(img)
    width = 500
    height = 500
    # remove noise
    img = obj.noise_removal(filterSize=(5, 5))
    prototxt = "../Deep-Learning-based-Edge-Detection/deploy.prototxt"
    caffemodel = "../Deep-Learning-based-Edge-Detection/hed_pretrained_bsds.caffemodel"
    cv.dnn_registerLayer('Crop', CropLayer)
    net = cv.dnn.readNet(prototxt, caffemodel)
    inp = cv.dnn.blobFromImage(img, scalefactor=1.0, size=(width, height),
                               mean=(104.00698793, 116.66876762, 122.67891434),
                               swapRB=False, crop=False)
    net.setInput(inp)
    out = net.forward()
    out = out[0, 0]
    out = cv.resize(out, (img.shape[1], img.shape[0]))
    out = 255 * out
    out = out.astype(np.uint8)
    # visualize
    cv.imshow("HED", out)
    cv.imshow("original", img)
    k = cv.waitKey(0) & 0xFF
    if(k == 27):
        cv.destroyAllWindows()