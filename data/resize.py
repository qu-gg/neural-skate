from scipy import misc
import imageio
import os
import numpy
from PIL import Image


resize_size = 128
out_path = "{}-set/".format(resize_size)
path = "decks/"

print("Resizing images...")
i = 0
for image_paht in os.listdir(path):
    input_path = os.path.join(path, image_paht)
    image = Image.open(input_path)
    resized = image.resize((resize_size, resize_size))
    imageio.imsave(out_path + str(i) + ".jpg", resized)

    #
    print("Image - ", i)
    i += 1

print("...complete!")
