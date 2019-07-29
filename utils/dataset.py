import argparse
import os
import glob
import cv2
from functools import partial
from tqdm import tqdm



def pad_and_crop_images(image, size_x, size_y, step_x, step_y, margin_color):

    # should the value of border be equal to background ?
    # padding
    image_y, image_x = image.shape[:2]
    border_y = 0
    if image_y % size_y != 0:
        border_y = (size_y - (image_y % size_y) + 1) // 2
        image = cv2.copyMakeBorder(image, border_y, border_y, 0, 0, cv2.BORDER_CONSTANT, value=margin_color)
        image_y = image.shape[0]
    border_x = 0
    if image_x % size_x != 0:
        border_x = (size_x - (image_x % size_x) + 1) // 2
        image = cv2.copyMakeBorder(image, 0, 0, border_x, border_x, cv2.BORDER_CONSTANT, value=margin_color)
        image_x = image.shape[1]

    # cropping
    # cannot adopt for loop, otherwise the cropped images will include very little margin
    patches = []
    start_y = 0
    while (start_y + size_y) <= image_y:
        start_x = 0
        while(start_x + size_x) <= image_x:
            patches.append(image[start_y:start_y + size_y, start_x:start_x + size_x])
            start_x += step_x
        start_y += step_y
    return patches, border_y, border_x

def save_images(image_patch_list, gt_patch_list, output_path):

    # save_path -> image_save_path os.path.join(args.output, 'image')
    #           -> gt_save_path    os.path.join(args.output, 'gt')
    # filename format: 'train' + '_id' + '.png'
    #                  'gt' + '_id' + '.png'

    image_patchpath = os.path.join(output_path, 'images')
    gt_patchpath = os.path.join(output_path, 'gt')

    for i, image in enumerate(image_patch_list):
        cv2.imwrite(os.path.join(image_patchpath, 'train' + str(i) + '.png'), image)
    for i, gt in enumerate(gt_patch_list):
        cv2.imwrite(os.path.join(gt_patchpath, 'gt' + str(i) + '.png'), gt)

def process_image(image_path, gt_path, size_x, size_y, step_x, step_y, image_margin_color, label_margin_color, output_path):

    # Read train and ground_truth images, cropping them and save

    #image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)
    image = cv2.imread(image_path)
    img_patches_list, _, _ = pad_and_crop_images(image, size_x, size_y, step_x, step_y, image_margin_color)
    #gt = cv2.cvtColor(cv2.imread(gt_path), cv2.COLOR_BGR2GRAY)
    gt = cv2.imread(gt_path)
    gt_patches_list, _, _ = pad_and_crop_images(gt, size_x, size_y, step_x, step_y, label_margin_color)

    assert len(img_patches_list) == len(gt_patches_list), \
        'MISMATCH ERROR, image and ground truth'

    save_images(img_patches_list, gt_patches_list, output_path)

def ensure_and_mkdir(path):

    if not os.path.exists(path):
        os.makedirs(path)

def parse_args():

    parser = argparse.ArgumentParser(description='configurations of dataset')

    parser.add_argument('-i', '--input', type=str, default=os.path.join('.','input'),
                        help='directory of input images, including images used to train and predict')
    parser.add_argument('-o', '--output', type=str, default=os.path.join('.','output'),
                        help='directory of output images')
    # whether shuffle or not is set in DataLoader
    parser.add_argument('--size_x', type=int, default=224,
                        help='the width of image patches')
    parser.add_argument('--size_y', type=str, default=224,
                        help='the height of image patches')
    # step is set equal to patch size by default, that is the overlap is zero
    # overlap = size - step
    parser.add_argument('--step_x', type=int, default=224,
                        help='the horizontal step of cropping images')
    parser.add_argument('--step_y', type=int, default=224,
                        help='the vertical step of cropping images')
    parser.add_argument('--image_margin_color', type=list, default=[255,255,255],
                        help='the color of image margin color')
    parser.add_argument('--label_margin_color', type=list, default=[255,255,255],
                        help='the color of label margin color')

    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()

    # confirm the directory of output images and gt
    ensure_and_mkdir(os.path.join(args.output))
    ensure_and_mkdir(os.path.join(args.output, 'images'))
    ensure_and_mkdir(os.path.join(args.output, 'gt'))

    # input images and ground truth
    # input image path: args.input/images
    # input gt path: args.input/gt

    image_pathes = glob.glob(os.path.join(args.input, 'images', '*'))
    gt_pathes = glob.glob(os.path.join(args.input, 'gt', '*'))

    f = partial(process_image, size_x=args.size_x, size_y=args.size_y, step_x=args.step_x, step_y=args.step_y,
                image_margin_color=args.image_margin_color, label_margin_color=args.label_margin_color, output_path=args.output)
    for index, (image_path, gt_path) in tqdm(enumerate(zip(image_pathes, gt_pathes))):
        f(image_path, gt_path)



