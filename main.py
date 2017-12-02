import numpy as np
import cv2
import sys
import os


def image_segmentation(input, K, output):
    img = cv2.imread(input)

    Z = img.reshape((-1, 2))
    Z = np.float32(Z)

    # type, max_iter = 25, epsilon = 1.0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 1.0)

    flags = cv2.KMEANS_RANDOM_CENTERS

    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, flags)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))


    cv2.imwrite(os.path.join("ClusteredImages/"+output+".png"), res2)

    # TODO: Uncomment these lines to view output
    #cv2.imshow('res2', res2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if len(sys.argv) == 8:

    image_segmentation(str(sys.argv[1]),  int(sys.argv[4]), str(sys.argv[5]))
    print("1st Image Processing complete")

    image_segmentation(str(sys.argv[2]),  int(sys.argv[4]), str(sys.argv[6]))
    print("2nd Image Processing complete")

    image_segmentation(str(sys.argv[3]),  int(sys.argv[4]), str(sys.argv[7]))
    print("3rd Image Processing complete")

    print("-----------------------------------------------------------")
    print("Process Complete files stored in ClusteredImages directory.")
    print("-----------------------------------------------------------")

else:
    print("usage: main.py <imagefile1> <imagefile2> <imagefile3> <number of clusters> <outputfile1> <outputfile2> <outputfile3>")