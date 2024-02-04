# Use a pipeline as a high-level helper
from PIL import Image
from transformers import pipeline
import torchvision.transforms as T
import torch
import numpy


def tensor2pil(image):
    return Image.fromarray(numpy.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(numpy.uint8))


def pil2tensor(image):
    return torch.from_numpy(numpy.array(image).astype(numpy.float32) / 255.0).unsqueeze(0)


class NSFWDetection:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "score": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "round": 0.001,
                    # The value represeting the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
                    "display": "nsfw_threshold"}),
                "alternative_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "run"

    CATEGORY = "NSFWDetection"

    def run(self, image, score, alternative_image):
        transform = T.ToPILImage()
        classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection")
        for i in range(len(image)):
            result = classifier(transform(image[i].permute(2, 0, 1)))
            image_size = image[i].size()
            width, height = image_size[1], image_size[0]
            for r in result:
                if r["label"] == "nsfw":
                    if r["score"] > score:
                        image[i] = pil2tensor(transform(alternative_image[0].permute(2, 0, 1)).resize((width, height),
                                                                               resample=Image.Resampling(2)))

        return (image,)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "NSFWDetection": NSFWDetection
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "NSFWDetection": "NSFW Detection"
}
