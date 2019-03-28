import torch
import numpy as np

'''
    Tensor has no attr 'copy' should use 'clone'
    pred 's requires_grad = True
    .clone().cpu().numpy()
'''
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
