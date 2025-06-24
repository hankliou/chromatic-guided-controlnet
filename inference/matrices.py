import numpy as np
import torch
import lpips
import math
import json
import cv2

# dubug usage
# import debugpy

# debugpy.listen(("0.0.0.0", 7860))
# print("Waiting for client to attach...")
# debugpy.wait_for_client()

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

def calculate_lpips(img1, img2):
    # 加载预训练的LPIPS模型
    lpips_model = lpips.LPIPS(net="alex")

    # 将图像转换为PyTorch的Tensor格式
    image1_tensor = torch.tensor(img1).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    image2_tensor = torch.tensor(img2).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    # 使用LPIPS模型计算距离
    return lpips_model(image1_tensor, image2_tensor).item()

def gen_txt_filename(name):
    global txt_psnr, txt_ssim, txt_lpips
    txt_psnr = f"psnr_{name}.txt" if not dddp_flag else f"psnr_{name}{dddp_testee}.txt"
    txt_ssim = f"ssim_{name}.txt" if not dddp_flag else f"ssim_{name}{dddp_testee}.txt"
    txt_lpips = f"lpips_{name}.txt" if not dddp_flag else f"lpips_{name}{dddp_testee}.txt"

# BSD
def bsd_loader():
    sample_path = "./output/BSD_B_Centroid_lock(probably)/testing/source"
    gt_path_replace = "../datasets/BSD_B_Centroid/testing/target"
    json_path = '../datasets/BSD_B_Centroid/testing/prompt.json'

    gen_txt_filename(sample_path.split('/')[2])

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
    sample_path = "./output/RealBlur-J_by_BSD_bothside_atten_1e-4_epoch103/"
    gt_path_replace = "RealBlur-J_ECC_IMCORR_centroid_itensity_ref/"
    json_path = '../datasets/RealBlur/RealBlur_J_test_list.json'

    gen_txt_filename(sample_path.split('/')[2])

    with open(json_path, 'rt') as js:
        for line in js:
            data = json.loads(line)
            gt.append('../datasets/RealBlur/' + data['target'])
            sample.append(data['target'].replace(gt_path_replace, sample_path, 1).replace('gt', 'blur'))
    sample.sort()
    gt.sort()
    
# GOPRO
def gopro_loader():
    sample_path = "./output/GOPRO_Large_augmented_512*512_epoch116/test/"
    gt_path_replace = "../datasets/GOPRO_Large/test/"
    json_path = '../datasets/GOPRO_Large/test/prompt.json'

    gen_txt_filename(sample_path.split('/')[2])

    with open(json_path, 'rt') as js:
        for line in js:
            data = json.loads(line)
            gt.append(data['target'])
            sample.append(data['target'].replace(gt_path_replace, sample_path, 1).replace('sharp', 'blur'))
    sample.sort()
    gt.sort()
    
    return sample, gt

# dddp
def dddp_loader():
    global dddp_flag, dddp_testee, dddp_indoor_list, dddp_outdoor_list
    dddp_flag = True
    dddp_testee = '_outdoor' # ['_indoor', '_outdoor', '']

    dddp_indoor_list = np.load("../datasets/dd_dp_dataset_png/indoor_labels.npy").tolist()
    dddp_outdoor_list = np.load("../datasets/dd_dp_dataset_png/outdoor_labels.npy").tolist()

    sample_path = "./output/dd_dp_dataset_png_lock_randomrop_512*512*2_ssimloss5-5_epoch675/"
    gt_path_replace = "../datasets/dd_dp_dataset_png/"
    json_path = '../datasets/dd_dp_dataset_png/test_prompt.json'

    gen_txt_filename(sample_path.split('/')[2])

    with open(json_path, 'rt') as js:
        js = json.load(js)
        for data in js:
            gt.append(data['target'])
            sample.append(data['source'].replace(gt_path_replace, sample_path, 1))
    
    return sample, gt

# init
average_psnr, average_ssim, average_lpips = 0, 0, 0
txt_psnr, txt_ssim, txt_lpips = '', '', ''
sample, gt = [], []
cnt, idx = 0, 0
flags = {
    'psnr' : True,
    'ssim' : True,
    'lpips' : True
}
error = False
dddp_flag = False
dddp_testee = ''
dddp_outdoor_list, dddp_indoor_list = [], []

# get value
# bsd_loader() # for bsd centroid only
gopro_loader() # for gopro only
# realblur_loader() # for realblur only
# dddp_loader() # for dd_dp only

# open file
txt_psnr = open(txt_psnr, 'w') if flags['psnr'] else None
txt_ssim = open(txt_ssim, 'w') if flags['ssim'] else None
txt_lpips = open(txt_lpips, 'w') if flags['lpips'] else None

for fname1, fname2 in zip(sample, gt):

    img1 = cv2.imread(fname1)
    img2 = cv2.imread(fname2)
    idx += 1
    if dddp_flag:
        if dddp_testee == '_outdoor' and idx in dddp_indoor_list: continue
        elif dddp_testee == '_indoor' and idx in dddp_outdoor_list: continue

    print(idx, fname1, fname2)


    # if error, flag error
    # if img1.all() == None or img2.all() == None:
    #     error = True
    #     continue
    
    # if size different, resize
    if img1.shape != img2.shape:
        img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]), interpolation=cv2.INTER_AREA)

    i1_array = np.array(img1)
    i2_array = np.array(img2)

    if flags['psnr']:
        r12 = psnr(i1_array, i2_array)
        txt_psnr.write(fname1 + " " + str(r12) + "\n")
        average_psnr += r12
    
    if flags['ssim']:
        r12 = ssim(i1_array, i2_array)
        txt_ssim.write(fname1 + " " + str(r12) + "\n")
        average_ssim += r12

    if flags['lpips']:
        r12 = calculate_lpips(i1_array, i2_array)
        txt_lpips.write(fname1 + " " + str(r12) + "\n")
        average_lpips += r12
        print(r12)
        
    cnt += 1
    
print('finish!')

if flags['psnr']:
    txt_psnr.write(f'avg psnr: {average_psnr / cnt}')
    if error: txt_psnr.write('error !!!!!!!!!!!!!!!') 
    txt_psnr.close()

if flags['ssim']:
    txt_ssim.write(f'avg ssim: {average_ssim / cnt}')
    if error: txt_ssim.write('error !!!!!!!!!!!!!!!') 
    txt_ssim.close()

if flags['lpips']:
    txt_lpips.write(f'avg lpips: {average_lpips / cnt}')
    if error: txt_lpips.write('error !!!!!!!!!!!!!!!') 
    txt_lpips.close()

print(f'psnr: {average_psnr / cnt}, ssim: {average_ssim / cnt}, lpips: {average_lpips / cnt}')
if error: 
    for i in range(30):
        print('error !!!!!!!!!!!!!!!') 