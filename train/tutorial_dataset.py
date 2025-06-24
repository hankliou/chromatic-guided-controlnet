import json
import cv2
import numpy as np

from torch.utils.data import Dataset

# # BSD_B_Centroid dataloader
# class MyDataset(Dataset):
#     def __init__(self):
#         self.data = []
#         with open('../datasets/BSD_B_Centroid/training/prompt.json', 'rt') as f:
#             for line in f:
#                 self.data.append(json.loads(line))

#     def __len__(self):
#         return len(self.data)
    
#     # random crop
#     def random_crop(self, source, target, color, crop_size):
#         import random
#         height, width, _ = source.shape
#         try:
#             left = random.randint(0, width - crop_size)
#             top = random.randint(0, height - crop_size)
#             right = left + crop_size
#             bottom = top + crop_size
            
#             crop_source = source[top:bottom, left:right]
#             crop_target = target[top:bottom, left:right]
#             crop_color = color[top:bottom, left:right]
            
#             return crop_source, crop_target, crop_color
#         except:
#             return cv2.resize(source, (512, 512)), cv2.resize(target, (512, 512)), cv2.resize(color, (512, 512))

#     def random_rotate(images):
#         """對一組圖片同步隨機旋轉 90, 180, 或 270 度"""
#         angle = np.random.choice([cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE])
#         return [cv2.rotate(img, angle) for img in images]

#     def __getitem__(self, idx):
#         item = self.data[idx]

#         source_filename = item['source']
#         target_filename = item['target']     
#         prompt_filename = item['prompt']   

#         source = cv2.imread(source_filename)
#         target = cv2.imread(target_filename)
#         prompt = cv2.imread(prompt_filename) # color block function        
        
#         # Do not forget that OpenCV read images in BGR order.
#         source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
#         target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
#         prompt = cv2.cvtColor(prompt, cv2.COLOR_BGR2RGB)
        
#         # Random crop --------------------
#         crop_size = 512
#         source, target, prompt = self.random_crop(source, target, prompt, crop_size)
#         # Normalization
#         source = source.astype(np.float32) / 255.0          # Normalize source images to [0, 1].        
#         target = (target.astype(np.float32) / 127.5) - 1.0  # Normalize target images to [-1, 1].
#         prompt = prompt.astype(np.float32) / 255.0          # Normalize source images to [0, 1].
        
#         # source, target, prompt = self.random_rotate([source, target, prompt])
        
#         return dict(jpg=target, hint=source, color_block=prompt)

# GoPro dataloader
class MyDataset(Dataset):
    def __init__(self):
        self.data = []
        with open('../datasets/GOPRO_Large/prompts.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)  # 2103     
    
    # random crop
    def random_crop(self, source, target, color, crop_size):
        import random
        height, width, _ = source.shape
        left = random.randint(0, width - crop_size)
        top = random.randint(0, height - crop_size)
        right = left + crop_size
        bottom = top + crop_size
        
        crop_source = source[top:bottom, left:right]
        crop_target = target[top:bottom, left:right]
        crop_color = color[top:bottom, left:right]
        
        return crop_source, crop_target, crop_color
    
    def random_rotate(images):
        """對一組圖片同步隨機旋轉 90, 180, 或 270 度"""
        angle = np.random.choice([cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE])
        return [cv2.rotate(img, angle) for img in images]

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt_filename = item['prompt']

        source = cv2.imread(source_filename)
        target = cv2.imread(target_filename)
        prompt = cv2.imread(prompt_filename) # color block function
        
        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        prompt = cv2.cvtColor(prompt, cv2.COLOR_BGR2RGB)
        
        # Random crop --------------------
        crop_size = 512
        source, target, prompt = self.random_crop(source, target, prompt, crop_size)
        # Normalization
        source = source.astype(np.float32) / 255.0          # Normalize source images to [0, 1].        
        target = (target.astype(np.float32) / 127.5) - 1.0  # Normalize target images to [-1, 1].
        prompt = prompt.astype(np.float32) / 255.0          # Normalize source images to [0, 1].
        
        # source, target, prompt = self.random_rotate([source, target, prompt])
        
        return dict(jpg=target, hint=source, color_block=prompt)

# # Realblur dataloader
# class MyDataset(Dataset):
#     def __init__(self):
#         self.data = []
#         with open('../datasets/RealBlur/RealBlur_J_train_list.json', 'rt') as f:
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
#         # source, target, prompt = self.random_crop(source, target, prompt, random_crop_size)
        
#         # Normalization
#         source = source.astype(np.float32) / 255.0          # Normalize source images to [0, 1].        
#         target = (target.astype(np.float32) / 127.5) - 1.0  # Normalize target images to [-1, 1].
#         prompt = prompt.astype(np.float32) / 255.0          # Normalize source images to [0, 1].

#         # return dict(jpg=target, txt=prompt, hint=source)
#         return dict(jpg=target, hint=source, color_block=prompt)

# # DPDD dataloader
# class MyDataset(Dataset):
#     def __init__(self):
#         self.data = []
#         with open('../datasets/dd_dp_dataset_png/train_prompt.json', 'rt') as f:
#             self.data = json.load(f)

#     def __len__(self):
#         return len(self.data) # 350
    
#     # histogram patches
#     def split_image_into_blocks(self, image, num_blocks):
#         height, width = image.shape[:2]
#         block_height = height // num_blocks
#         block_width = width // num_blocks
#         blocks = []
#         for i in range(num_blocks):
#             for j in range(num_blocks):
#                 block = image[i * block_height: (i + 1) * block_height, j * block_width: (j + 1) * block_width]
#                 blocks.append(block)
#         return blocks        
    
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
        
#         source_filename = item['source']
#         target_filename = item['target']
#         prompt_filename = item['prompt']

#         source = cv2.imread(source_filename)
#         target = cv2.imread(target_filename)
#         prompt = cv2.imread(prompt_filename) # color block function
        
#         # Do not forget that OpenCV read images in BGR order.
#         source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
#         target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
#         prompt = cv2.cvtColor(prompt, cv2.COLOR_BGR2RGB)
        
#         # Random crop --------------------
#         ret = []
#         crop_size = 512
#         crop_quan = 2
#         for _ in range(crop_quan):
#             source, target, prompt = self.random_crop(source, target, prompt, crop_size)
#             # Normalization
#             source = source.astype(np.float32) / 255.0          # Normalize source images to [0, 1].        
#             target = (target.astype(np.float32) / 127.5) - 1.0  # Normalize target images to [-1, 1].
#             prompt = prompt.astype(np.float32) / 255.0          # Normalize source images to [0, 1].
#             ret.append(dict(jpg=target, hint=source, color_block=prompt))
#         return ret
