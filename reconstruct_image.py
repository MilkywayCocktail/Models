import numpy as np
import matplotlib.pyplot as plt
import cv2


def reconstruct(bbx_path, mask_path, depth_path):
    bbx = np.load(bbx_path)
    mask = np.load(mask_path)
    depth = np.load(depth_path)

    # Restore bbx
    bbx[..., -1] = bbx[..., -1] - bbx[..., -3]
    bbx[..., -2] = bbx[..., -2] - bbx[..., -4]

    img_size = (128, 226)
    imgs = np.zeros((len(bbx), *img_size))
    min_area = 0

    for i in range(len(bbx)):
        img = np.squeeze(mask[i]).astype('float32')
        (T, timg) = cv2.threshold((img * 255).astype(np.uint8), 1, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(timg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) != 0:
            contour = max(contours, key=lambda x: cv2.contourArea(x))
            area = cv2.contourArea(contour)

            if area < min_area:
                # print(area)
                pass

            else:
                x, y, w, h = cv2.boundingRect(contour)
                subject = img[y:y + h, x:x + w] * np.squeeze(depth[i])

                x1, y1, w1, h1 = bbx[i]
                imgs[y1:y1+h, x1:x1+h] = cv2.resize(subject, (w1, h1))
