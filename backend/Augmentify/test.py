from PIL import Image
from Augmentify.augment.geometric import rotate

img = Image.open("example.jpg")
rotated_img = rotate(img)
rotated_img.show()