import Algorithmia
import cv2

def apply(input):
    return "hello {}".format(input) + "opencv version: " + cv2.__version__
