import numpy as np
from PIL import Image
import torch
import cv2 as cv

# Tensor to PIL


def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# PIL to Tensor


def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class ImageMedianFilter:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "diameter": ("INT", {"default": 20}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_median_filter"
    CATEGORY = "Tools"

    def medianFilter(self, img, diameter):
        # Convert diameter to an odd integer
        diameter = int(diameter)
        if diameter % 2 == 0:
            diameter += 1  # Ensure the diameter is odd

        # Convert the image to RGB if it's not already
        img = img.convert('RGB')

        # Convert PIL image to numpy array and then to BGR format for OpenCV
        img = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)

        # Apply median filter
        img = cv.medianBlur(img, diameter)

        # Convert the image back to RGB format
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        # Convert numpy array back to PIL image
        return Image.fromarray(img).convert('RGB')

    def apply_median_filter(self, image, diameter):
        tensors = []
        if len(image) > 1:
            print('aaaaa')
            for img in image:

                pil_image = None
                # PIL Image
                pil_image = tensor2pil(img)

                # Apply Median Fliter
                new_img = self.medianFilter(pil_image, diameter)

                # Output image
                out_image = (pil2tensor(new_img) if pil_image else img)

                tensors.append(out_image)

            tensors = torch.cat(tensors, dim=0)

        else:
            print('bbbb')
            pil_image = None
            img = image
            # PIL Image
            
            pil_image = tensor2pil(img)
            # Apply Median Fliter
            new_img = self.medianFilter(pil_image, diameter)

            # Output image
            out_image = (pil2tensor(new_img) if pil_image else img)

            tensors = out_image

        return (tensors, )


# Set the web directory, any .js file in that directory will be loaded by the frontend as a frontend extension
# WEB_DIRECTORY = "./somejs"


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "ImageMedianFilter": ImageMedianFilter
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageMedianFilter": "Apply Median Filter"
}
