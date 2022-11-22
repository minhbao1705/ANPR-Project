import numpy as np
import torch
from char_reg.Model import Lenet
import cv2

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

ALPHA_DICT = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'K', 9: 'L', 10: 'M', 11: 'N', 12: 'P',
              13: 'R', 14: 'S', 15: 'T', 16: 'U', 17: 'V', 18: 'X', 19: 'Y', 20: 'Z', 21: '0', 22: '1', 23: '2',
              24: '3',
              25: '4', 26: '5', 27: '6', 28: '7', 29: '8', 30: '9', 31: "Background"}


def load_model(path):
    model = Lenet()
    model.load_state_dict(torch.load(path, map_location=device))
    # model.to(device)
    model.eval()
    return model


def swap(lis):
    return [(sub[1], sub[0]) for sub in lis]


def check_box(H, W, thresh):
    if (H / W <= thresh):  # 1line lp
        return 1
    return 2  # 2 line lp


def convertToChar(predict):
    return ALPHA_DICT[int(predict[0].numpy())]

def sort_label(dict_predict, type):  # dict{(x,y):label, ...}
    result = ""
    if (len(dict_predict.keys()) != 0):
        if type == 1:
            keys = [key for key in sorted(dict_predict.keys())]  # [(x,y),...]
            for x in keys:
                result += dict_predict[x]  # append label
        if type == 2:

            temp = [[], []]
            keys = swap(dict_predict.keys())
            keys = sorted(keys)

            maxkey = keys[-1][0]
            minkey = keys[0][0]
            avg_thresh = minkey + (maxkey - minkey) / 3

            keys = swap(keys)
            for i in keys:
                if (i[1] <= avg_thresh):
                    temp[0].append(i)
                else:
                    temp[1].append(i)
            for i in range(len(temp)):
                temp[i] = sorted(temp[i])
            for i in range(len(temp)):
                for coords in temp[i]:
                    result += dict_predict[coords]
    return result


def segmentation(model, img):
    with torch.no_grad():
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        cv2.imshow("", thresh)
        cv2.waitKey(0)
        output = cv2.connectedComponentsWithStats(
            thresh, 4, cv2.CV_32S)
        (numLabels, labels, stats, centroids) = output

        bbox_h = img.shape[0]
        bbox_w = img.shape[1]
        bbox_area = bbox_h * bbox_w

        type_lp = check_box(bbox_h, bbox_w, 0.3)
        val_w = 0.03
        val_h = 0.03
        val_area_a = 0.001

        dict_predict = {}
        for i in range(1, numLabels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]

            minwidth = w >= 2
            minheight = h >= 4
            aspectRatio = (w / h >= 0.01) and (w / h <= 1)
            keepWidth = w / bbox_w >= (val_w) and w / bbox_w <= (val_w + 0.47)
            keepHeight = h / bbox_h >= (val_h) and h / bbox_h <= (val_h + 0.57)
            keepArea = area / bbox_area >= val_area_a and area / bbox_area <= (val_area_a + 0.1)


            if all((minwidth, minheight, aspectRatio, keepWidth, keepHeight, keepArea)):
                seg = gray[y:y + h, x:x + w]
                cv2.imshow("", seg)
                cv2.waitKey(0)
                resized = np.array([cv2.resize(seg, (32, 32), interpolation=cv2.INTER_NEAREST)]) / 255
                resized = torch.from_numpy(resized).unsqueeze(0)

                output = model(resized.float())
                print(output)
                _, predicted = torch.max(output, 1)
                # print(int(predicted[0].numpy()))
                if convertToChar(predicted) != "Background":
                    dict_predict[(x, y)] = convertToChar(predicted)
    return sort_label(dict_predict, type_lp)

