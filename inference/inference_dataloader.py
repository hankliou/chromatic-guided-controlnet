import json
import cv2
import torch
import numpy as np

from torch.utils.data import Dataset

# bsd / gopro dataloader
class MyDataset(Dataset):
    def __init__(self):
        self.data = []
        with open(
            # "../datasets/GOPRO_Large/test/prompt.json", "rt"
            "../datasets/GOPRO_Large/test/single_inference.json"
            # "../datasets/BSD_B_Centroid/testing/prompt.json", "rt"
        ) as f:  # path to test data
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item["source"]
        target_filename = item["target"]
        prompt_filename = item["prompt"]

        source = cv2.imread(source_filename)
        target = cv2.imread(target_filename)
        prompt = cv2.imread(prompt_filename)

        source = cv2.resize(source, (1280, 768))
        target = cv2.resize(target, (1280, 768))
        prompt = cv2.resize(prompt, (1280, 768))

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        prompt = cv2.cvtColor(prompt, cv2.COLOR_BGR2RGB)

        source = source.astype(np.float32) / 255.0          # Normalize source images to [0, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0  # Normalize target images to [-1, 1].
        prompt = prompt.astype(np.float32) / 255.0

        # convert to tensor
        target = torch.from_numpy(target)
        source = torch.from_numpy(source)
        prompt = torch.from_numpy(prompt)

        # modify, remove 'txt' field
        return dict(
            jpg=target,
            hint=source,
            color_block=prompt,
            file_name='/'.join(item["source"].split("/")[2:]),
        )
    
# # dd_dp dataloader
# class MyDataset(Dataset):
#     def __init__(self):
#         self.data = []
#         with open('../datasets/dd_dp_dataset_png/test_prompt.json', 'rt') as f:
#             self.data = json.load(f)

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         item = self.data[idx]

#         source_filename = item["source"]
#         target_filename = item["target"]
#         prompt_filename = item["prompt"]

#         source = cv2.imread(source_filename)
#         target = cv2.imread(target_filename)
#         prompt = cv2.imread(prompt_filename)

#         # resize
#         source = cv2.resize(source, (1216, 832))
#         target = cv2.resize(target, (1216, 832))
#         prompt = cv2.resize(prompt, (1216, 832))

#         # Do not forget that OpenCV read images in BGR order.
#         source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
#         target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
#         prompt = cv2.cvtColor(prompt, cv2.COLOR_BGR2RGB)

#         source = source.astype(np.float32) / 255.0          # Normalize source images to [0, 1].
#         target = (target.astype(np.float32) / 127.5) - 1.0  # Normalize target images to [-1, 1].
#         prompt = prompt.astype(np.float32) / 255.0

#         # convert to tensor
#         target = torch.from_numpy(target)
#         source = torch.from_numpy(source)
#         prompt = torch.from_numpy(prompt)

#         # modify, remove 'txt' field
#         return dict(
#             jpg=target,
#             hint=source,
#             color_block=prompt,
#             file_name='/'.join(item["source"].split("/")[2:]),
#         )

# # Realblur dataloader
# class MyDataset(Dataset):
#     def __init__(self):
#         self.data = []
#         with open('../datasets/RealBlur/RealBlur_J_test_list.json', 'rt') as f:
#             for line in f:
#                 self.data.append(json.loads(line))

#     def __len__(self):
#         return len(self.data)  
    
#     # random crop
#     def random_crop(self, source, target, color, crop_size):
#         import random
#         height, width, _ = source.shape
#         left = random.randint(0, width - crop_size)
#         top = random.randint(0, height - crop_size)
#         right = left + crop_size
#         bottom = top + crop_size
        
#         crop_source = source[top:bottom, left:right]
#         crop_target = target[top:bottom, left:right]
#         crop_color = color[top:bottom, left:right]
        
#         return crop_source, crop_target, crop_color

#     def __getitem__(self, idx):
#         item = self.data[idx]

#         source_filename = '../datasets/RealBlur/' + item['source']
#         target_filename = '../datasets/RealBlur/' + item['target']
#         prompt_filename = '../datasets/RealBlur/' + item['prompt']

#         source = cv2.imread(source_filename)
#         target = cv2.imread(target_filename)
#         prompt = cv2.imread(prompt_filename) # color block function
        
#         # resize (don't resize if random crop)
#         source = cv2.resize(source, (512, 512))
#         target = cv2.resize(target, (512, 512))
#         prompt = cv2.resize(prompt, (512, 512))
        
#         # Do not forget that OpenCV read images in BGR order.
#         source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
#         target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
#         prompt = cv2.cvtColor(prompt, cv2.COLOR_BGR2RGB)
        
#         # Random crop
#         # random_crop_size = 128
#         # source, target, prompt = self.random_crop(source, target, prompt, random_crop_size)
        
#         # Normalization
#         source = source.astype(np.float32) / 255.0          # Normalize source images to [0, 1].        
#         target = (target.astype(np.float32) / 127.5) - 1.0  # Normalize target images to [-1, 1].
#         prompt = prompt.astype(np.float32) / 255.0          # Normalize source images to [0, 1].

#         # modify, remove 'txt' field
#         return dict(
#             jpg=target,
#             hint=source,
#             color_block=prompt,
#             file_name='/' + item["source"],
#         )