from PIL import Image
import cv2
import sys

WIDTH = 310/2
HEIGHT = 568/2

size = (int(WIDTH), int(HEIGHT))

def resize_and_save(img_name):
    try:
        main_image = Image.open("images/{}".format(img_name))
    except IOError:
        sys.stderr.write("ERROR: Could not open file {}\n".format(img_name))
        exit(1)
        
    #main_image.thumbnail(size, Image.ANTIALIAS)
    x= main_image.resize(size, Image.NEAREST)
    x.save("images/resized/{}".format(img_name))

if __name__=="__main__":
    resize_and_save("01.jpg")
