from scipy import misc
import os
import numpy

resize_size = 64
out_path = "{}-set/".format(resize_size)
path = "decks/"

print("Resizing images...")
i = 0
for image_paht in os.listdir(path):
    input_path = os.path.join(path, image_paht)
    image = misc.imread(input_path)
    resized = misc.imresize(image, [resize_size, resize_size])
    misc.imsave(out_path + str(i) + ".jpg", resized)
    print("Image - ", i)
    i += 1

print("...complete!")
