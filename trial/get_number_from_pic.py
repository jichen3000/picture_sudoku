import Image
import os

if __name__ == '__main__':
    from minitest import *

    pic_path = "../resource/example_pics"
    original_image_name = "original.jpg"
    original_gray_image_name = "original_gray.jpg"
    
    with test("as array"):
        original_image = Image.open(os.path.join(pic_path,original_image_name)) # open colour image
        # original_gray_image = original_image.convert('1') # convert image to black and white
        # original_gray_image.save(os.path.join(pic_path,original_gray_image_name))

        original_gray_image = original_image.convert('L')
        # original_gray_image = original_gray_image.point(lambda x: 0 if x<128 else 255, '1')
        original_gray_image.save(os.path.join(pic_path,original_gray_image_name))


