import numpy as np
import PIL
from PIL import Image
from torchvision import datasets

data_train = datasets.MNIST(root = "../Dataset/myMNIST/",
                            train = True,
                            download = True)