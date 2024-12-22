import argparse
import stippler
import edge

import cv2
import numpy as np

description = ""
parser = argparse.ArgumentParser(description=description)
parser.add_argument('filename', metavar='image filename', type=str,
                    help='filename')
args = parser.parse_args()

filename = args.filename

original_image = cv2.imread(filename)
 
#Accessing height, width and channels
# Height of the image
height = original_image.shape[0]
# Width of the image
width = original_image.shape[1]

#get edge and stipple image
stippler.run_stippler(filename)
edge.detect_edge(filename)

#get file
stippler_image = cv2.imread("stipple.png")
edge_image = cv2.imread("edge.png")

stippler_image = cv2.resize(stippler_image, (width, height))
edge_image = cv2.resize(edge_image, (width, height))

combine_image = cv2.add(stippler_image, edge_image)

cv2.imwrite("combined_image.png", combine_image)

print("finished")



