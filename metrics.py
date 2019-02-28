import torch
import numpy as np

'''
    Tensor has no attr 'copy' should use 'clone'
    pred 's requires_grad = True
    .clone().cpu().numpy()
'''
def Accuracy1(pred, label):

    pred = pred.detach().cpu().numpy()
    pred = np.argmax(pred, axis=1)
    label = label.cpu().numpy()
    # ignore index 0 background return True False map
    valid = (label > 0)
    # valid and accuracy pixel.sum()
    acc_sum = (valid * (pred == label)).sum()
    # valid pixel sum
    pixel_sum = valid.sum()
    # epsilon
    acc = acc_sum / (pixel_sum +1e-10)
    return acc

def Accuracy(pred, label):

    with torch.no_grad():
        pred = torch.argmax(pred, dim=1)
        pred = pred.view(-1)
        label = label.view(-1)
        # ignore 0 background
        valid = (label > 0).long()
        # convert to float() 做除法的时候分子和分母都要转换成 float 如果是long 则会出现zero
        # .long() convert boolean to long then .float() convert to float
        # 合法的 pred == label的 pixel总数
        acc_sum = torch.sum(valid * (pred == label).long()).float()
        # 合法的pixel总数
        pixel_sum = torch.sum(valid).float()
        # epsilon
        acc = acc_sum / (pixel_sum + 1e-10)
        return acc

def MIoU1(pred, label, nb_classes):

    pred = pred.detach().cpu().numpy()
    pred = np.argmax(pred, axis=1)
    label = label.cpu().numpy()
    #pred = np.asarray(pred).copy()
    #label = np.asarray(label).copy()
    #label = label.clone()
    pred += 1
    label +=1

    # ignore index 0 background    --> change to 1
    pred = pred * (label > 0)
    #print(pred)
    intersection = pred * (pred == label)
    #print(intersection)
    (area_interection, _) = np.histogram(intersection, bins=nb_classes,
                                         range=(1, nb_classes))
    (area_pred, _) = np.histogram(pred, bins=nb_classes, range=(1, nb_classes))
    (area_lab, _) = np.histogram(label, bins=nb_classes, range=(1, nb_classes))
    area_union = area_pred + area_lab - area_interection

    miou = area_interection/ area_union
    mean_iou = miou[1:].sum() / (nb_classes-1)
    return mean_iou

def MIoU(pred, label, nb_classes):

    with torch.no_grad():
        pred = torch.argmax(pred, dim=1)
        pred = pred.view(-1)
        label = label.view(-1)
        iou = torch.zeros(nb_classes ).to(pred.device)
        for k in range(1, nb_classes):
            # pred_inds ,target_inds boolean map
            pred_inds = pred == k
            target_inds = label == k
            intersection = pred_inds[target_inds].long().sum().float()
            union = (pred_inds.long().sum() + target_inds.long().sum() - intersection).float()

            iou[k] = (intersection/ (union+1e-10))

        return (iou.sum()/ (nb_classes-1))


if __name__ == '__main__':
    a = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                  [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                  [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                  [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                  [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                  [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                  [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],])
    b = np.array([[1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                  [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                  [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                  [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                  [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                  [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                  [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],])

    a = torch.from_numpy(a).long()
    b = torch.from_numpy(b).long()

    # intersection = 16, union =
    acc = Accuracy(a,b)
    print(acc)
