import os
import torch
import einops
import config
import numpy as np
from share import *
from PIL import Image
from torch.utils.data import DataLoader
from cldm.ddim_hacked import DDIMSampler
from inference_dataloader import MyDataset
from cldm.model import create_model, load_state_dict

# dubug usage
# import debugpy

# debugpy.listen(("0.0.0.0", 7860))
# print("Waiting for client to attach...")
# debugpy.wait_for_client()

# Configs
checkpoint_path = "../checkpoints/BSD_bothside_attent_resize_1e-4/new_exp_sd21_epoch=103_step=103999.ckpt"
batch_size = 4

model = create_model("./models/cldm_v21.yaml").cpu()
model.load_state_dict(load_state_dict(checkpoint_path, location="cpu"))
model = model.cuda()
model.eval()
ddim_sampler = DDIMSampler(model)

# Misc
dataset = MyDataset()
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=False)

# Gerenate!
# if config.save_memory:
#     model.low_vram_shift(is_diffusing=True)

index = 0
pic_left = len(dataset)

ddim_steps = 50
strength = 1.0
guess_mode = False
count = 0
for x in dataloader:
    
    print('x: ', x.keys())

    if pic_left < batch_size:
        batch_size = pic_left
    else:
        pic_left -= batch_size
    # count += batch_size
    # if count > len(val_dataset.dataset):
    #     batch_size = batch_size - (count - len(val_dataset.dataset))

    with torch.no_grad():
        z, c = model.get_input(x, "", bs=batch_size)
        control = c["c_concat"][0][:batch_size]
        color_block = c["c_crossattn"][0][:batch_size]  #
        cond = {"c_concat": [control], "color_block": [color_block]}
        uc_full = {"c_concat": [control], "color_block": [color_block]}
        shape = (4, 512 // 8, 512 // 8)

        # if config.save_memory:
        #     model.low_vram_shift(is_diffusing=True)

        model.control_scales = (
            [strength * (0.825 ** float(12 - i)) for i in range(13)]
            if guess_mode
            else ([strength] * 13)
        )  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(
            ddim_steps,
            batch_size,
            shape,
            cond,
            verbose=False,
            eta=0.0,
            unconditional_guidance_scale=9.0,
            unconditional_conditioning=uc_full,
        )

        # if config.save_memory:
        #     model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (
            (einops.rearrange(x_samples, "b c h w -> b h w c") * 127.5 + 127.5)
            .cpu()
            .numpy()
            .clip(0, 255)
            .astype(np.uint8)
        )

        # results = [x_samples[i] for i in range(batch_size)]
        path = "./output"
        if not os.path.exists(path):
            os.makedirs(path)

        # for id, result in enumerate(results):
        for id, result in enumerate(x_samples):
            dir = '/'.join((path + f"/{x['file_name'][id]}").split('/')[:-1])
            if not os.path.isdir(dir):
                os.makedirs(dir)
            Image.fromarray(result).save(path + f"/{x['file_name'][id]}")
            print(path + f"/{x['file_name'][id]}")
            count += 1

print(f'count: {count}')