import numpy as np
from PIL import Image
import torch
import cv2

# Tensor to PIL


def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# PIL to Tensor


def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


class ImageKmeansFilter:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "k": ("INT", {"default": 18}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_kmeans_filter"
    CATEGORY = "Tools"

    def reduce_colors_kmeans(image, n_clusters=8):
        """
        使用 K-means 聚类算法减少图像颜色。

        Args:
        image: PIL Image 对象.
        n_clusters: 聚类数量，即最终图像的颜色数量.

        Returns:
        颜色减少后的 PIL Image 对象.
        """

        # 将 PIL Image 转换为 NumPy 数组
        img = np.array(image)

        # OpenCV 期望颜色通道顺序为 BGR，PIL 使用 RGB
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # 将图像转换为二维数组，每行代表一个像素
        pixels = img.reshape((-1, 3))

        # 将像素数据类型转换为 float32
        pixels = np.float32(pixels)

        # 定义 K-means 聚类参数
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

        # 执行 K-means 聚类
        _, labels, centers = cv2.kmeans(pixels, n_clusters, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

        # 将聚类中心转换为 uint8 类型
        centers = np.uint8(centers)

        # 使用聚类标签重建图像
        reduced_image = centers[labels.flatten()]

        # 将图像 reshape 回原始尺寸
        reduced_image = reduced_image.reshape(img.shape)

        # 将图像转换回 RGB 色彩空间
        reduced_image = cv2.cvtColor(reduced_image, cv2.COLOR_BGR2RGB)

        # 将 NumPy 数组转换回 PIL Image
        reduced_image = Image.fromarray(reduced_image)

        return reduced_image

    def apply_kmeans_filter(self, image, k):
        tensors = []
        if len(image) > 1:
            for img in image:

                pil_image = None
                # PIL Image
                pil_image = tensor2pil(img)

                # Apply Median Fliter
                new_img = self.reduce_colors_kmeans(pil_image, k)
 
                # Output image
                out_image = (pil2tensor(new_img) if pil_image else img)

                tensors.append(out_image)

            tensors = torch.cat(tensors, dim=0)

        else:
            pil_image = None
            img = image
            # PIL Image

            pil_image = tensor2pil(img)
            # Apply Median Fliter
            new_img = self.reduce_colors_kmeans(pil_image, k)

            # Output image
            out_image = (pil2tensor(new_img) if pil_image else img)

            tensors = out_image

        return (tensors, )


# Set the web directory, any .js file in that directory will be loaded by the frontend as a frontend extension
# WEB_DIRECTORY = "./somejs"


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "ImageKmeansFilter": ImageKmeansFilter
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageKmeansFilter": "Apply Kmeans Filter"
}
