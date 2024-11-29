from PIL import Image
import numpy as np
import math
import json
import cv2

def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()

def calculate_ssim(img1, img2):
    """calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    """
    if not img1.shape == img2.shape:
        raise ValueError("Input images must have the same dimensions.")
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError("Wrong input image dimensions.")

# BSD
def bsd_loader():
    sample_path = "./output/BSD_B_Centroid_lock(probably)/testing/source"
    gt_path_replace = "../datasets/BSD_B_Centroid/testing/target"
    json_path = '../datasets/BSD_B_Centroid/testing/prompt.json'

    global txt_psnr, txt_ssim
    txt_psnr = f"psnr_{sample_path.split('/')[2]}.txt"
    txt_ssim = f"ssim_{sample_path.split('/')[2]}.txt"

    with open(json_path, 'rt') as js:
        for line in js:
            data = json.loads(line)
            gt.append(data['target'])
            sample.append(data['target'].replace(gt_path_replace, sample_path, 1).replace('sharp', 'blur'))
    sample.sort()
    gt.sort()
    
    return sample, gt

# Realblur
def realblur_loader():
    sample_path = "./output/RealBlur-J_epoch_60/"
    gt_path_replace = "RealBlur-J_ECC_IMCORR_centroid_itensity_ref/"
    json_path = '../datasets/RealBlur/RealBlur_J_test_list.json'

    global txt_psnr, txt_ssim
    txt_psnr = f"psnr_{sample_path.split('/')[2]}.txt"
    txt_ssim = f"ssim_{sample_path.split('/')[2]}.txt"

    with open(json_path, 'rt') as js:
        for line in js:
            data = json.loads(line)
            gt.append('../datasets/RealBlur/' + data['target'])
            sample.append(data['target'].replace(gt_path_replace, sample_path, 1).replace('gt', 'blur'))
    sample.sort()
    gt.sort()
    
# GOPRO
def gopro_loader():
    sample_path = "./output/GOPRO_Large_gen_by_BSD_bothside_atten/test/"
    gt_path_replace = "../datasets/GOPRO_Large/test/"
    json_path = '../datasets/GOPRO_Large/test/prompt.json'

    global txt_psnr, txt_ssim
    txt_psnr = f"psnr_{sample_path.split('/')[2]}.txt"
    txt_ssim = f"ssim_{sample_path.split('/')[2]}.txt"

    with open(json_path, 'rt') as js:
        for line in js:
            data = json.loads(line)
            gt.append(data['target'])
            sample.append(data['target'].replace(gt_path_replace, sample_path, 1).replace('sharp', 'blur'))
    sample.sort()
    gt.sort()
    
    return sample, gt

# init
average_psnr = 0
average_ssim = 0
cnt = 0
cnt_ssim = 0
sample, gt = [], []
txt_psnr, txt_ssim = '', ''

# get value
# bsd_loader() # for bsd centroid only
gopro_loader() # for gopro only
# realblur_loader() # for realblur only

# open file
print(txt_psnr, txt_ssim)
txt_psnr = open(txt_psnr, 'w')
txt_ssim = open(txt_ssim, 'w')
    
for fname1, fname2 in zip(sample, gt):
    # print(fname1, fname2)
    img1 = cv2.imread(fname1, 0)
    img2 = cv2.imread(fname2, 0)
    
    # if size different, resize
    if img1.shape != img2.shape:
        img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]), interpolation=cv2.INTER_AREA)

    i1_array = np.array(img1)
    i2_array = np.array(img2)

    r12 = psnr(i1_array, i2_array)
    txt_psnr.write(fname1 + " " + str(r12) + "\n")
    average_psnr += r12
    
    r12 = ssim(i1_array, i2_array)
    txt_ssim.write(fname1 + " " + str(r12) + "\n")
    average_ssim += r12
    
    cnt += 1
    
    # if r12 < 27.5:
    #     cv2.imwrite(f'{bad_pic}{fname1}', img1)
    #     cv2.imwrite(f'{bad_pic}{fname2}', img2)
    # print(fname1, fname2, r12)
    
txt_psnr.write(f'avg psnr: {average_psnr / cnt}')
txt_psnr.close()

txt_ssim.write(f'avg psnr: {average_ssim / cnt}')
txt_psnr.close()

print(f'psnr: {average_psnr / cnt}, ssim: {average_ssim / cnt}')
