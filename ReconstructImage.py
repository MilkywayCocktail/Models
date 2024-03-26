import numpy as np
import matplotlib.pyplot as plt
import cv2


class Reconstruct:
    def __init__(self, name, path):
        self.name = name
        self.path = path
        self.preds: dict = np.load(self.path, allow_pickle=True).item()

        self.bbx = np.array(self.preds['S_BBX'])
        self.depth = np.array(self.preds['S_DPT'])
        self.mask = np.array(self.preds['S_PRED'])
        self.inds = self.preds['IND']

        print(f"Loaded bbx of {self.bbx.shape}, depth of {self.depth.shape}, mask of {self.mask.shape}")

        self.img_size = (128, 226)
        self.imgs = np.zeros((len(self.bbx), *self.mg_size))
        self.min_area = 0
        self.fail_count = 0

    def reconstruct(self):
        print("Reconstructing...", end='')
        for i in range(len(self.bbx)):
            img = np.squeeze(self.mask[i]).astype('float32')
            (T, timg) = cv2.threshold((img * 255).astype(np.uint8), 1, 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(timg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) != 0:
                contour = max(contours, key=lambda x: cv2.contourArea(x))
                area = cv2.contourArea(contour)

                if area < self.min_area:
                    # print(area)
                    pass

                else:
                    x, y, w, h = cv2.boundingRect(contour)
                    subject = img[y:y + h, x:x + w] * np.squeeze(self.depth[i])

                    x1, y1, x2, y2 = self.bbx[i]
                    x1 = int(x1 * 226)
                    y1 = int(y1 * 128)
                    x2 = int(x2 * 226)
                    y2 = int(y2 * 128)
                    w1 = x2 - x1
                    h1 = y2 - y1

                    try:
                        subject1 = cv2.resize(subject, (w1, h1))
                        for x in range(w1):
                            for y in range(h1):
                                self.imgs[i, y1 + y, x1 + x] = subject1[y, x]
                    except Exception as e:
                        print(e)
                        print(x1, y1, x2, y2, w1, h1)
                        print(subject1.shape)
                        self.fail_count += 1
        print("Done")
        print(f"Reconstruction finished. Failure count = {self.fail_count}")

    