import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def add_margin(pil_img, top, left, new_width):
    result = Image.new(pil_img.mode, (new_width, new_width))
    result = result.convert('L')
    result.paste(pil_img, (left, top))
    return result


def random_deviate(mn):
    shrink_rate = np.random.randint(3, 10, len(mn)) / 10
    re_size = (28 * shrink_rate).astype(np.int8)
    deviation = np.zeros((len(re_size), 2), dtype=int)
    value_rate = np.random.rand(len(mn))
    labels = np.array([re_size, value_rate, deviation[:, 0], deviation[:, 1]])

    for i in range(len(mn)):
        deviation[i] = np.random.randint(0, 29 - re_size[i], 2)
        im = Image.fromarray(mn[i], mode='L')
        im = im.resize((re_size[i], re_size[i]))
        im = add_margin(im, deviation[i, 0], deviation[i, 1], 28)
        im = (np.asarray(im) * value_rate[i]).astype(np.uint8)
        mn[i] = im

        plt.figure()
        plt.imshow(im)
        plt.colorbar()
        plt.show()

    return mn, labels


mnist_path = '../../dataset/myMNIST/train_images.npy'

mnist = np.load(mnist_path)

inds = np.random.choice(np.arange(60000), 10, replace=False)
inds.sort()

mni = mnist[inds]
mni = np.tile(mni, 2000)
mni = mni.reshape((-1, 28, 28))
print(mni.shape)

mni, labels = random_deviate(mni)
labels = labels.transpose()
print(labels.shape)
# np.save(mnist_path[:-4] + '_10c2000s.npy', mni.reshape((-1, 1, 28, 28)))
# np.save(mnist_path[:-4] + '_10c2000s_label.npy', labels)
