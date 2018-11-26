from scipy import misc
import os

out_path  = "grayscale/"
path = "decks/"

print("Refactoring images...")
i = 0
for image_paht in os.listdir(path):
    input_path = os.path.join(path, image_paht)
    grayscale = misc.imread(input_path, mode="L")
    misc.imsave(out_path + str(i) + ".jpg", grayscale)
    print("Image - ", i)
    i += 1

print("...complete!")
