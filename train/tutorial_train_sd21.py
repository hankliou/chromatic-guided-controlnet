from share import *
import math

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

# # dubug usage
# import debugpy
# debugpy.listen(("0.0.0.0", 7860))
# print("Waiting for client to attach...")
# debugpy.wait_for_client()

# Configs
resume_path = './models/v2-1_768-ema-pruned_ini_none.ckpt'
# resume_path = '../checkpoints/new_exp_sd21_epoch=212_step=110951.ckpt'
batch_size = 4
logger_freq = 300
learning_rate = 1e-4
sd_locked = True
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v21.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

# save middle ckpt for back up
# ================================================================================================
from pytorch_lightning.callbacks import ModelCheckpoint
import os
directory = "../checkpoints/"
if not os.path.exists(directory):
    os.makedirs(directory)

checkpoint_callback = ModelCheckpoint(dirpath = directory,
                                      save_top_k = -1,
                                      every_n_train_steps=math.ceil(16000/batch_size), save_last=True,
                                      save_weights_only=False,
                                      filename='new_exp_sd21_{epoch:02d}_{step:06d}')
# ================================================================================================

# Misc
dataset = MyDataset()
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger, checkpoint_callback])#, resume_from_checkpoint = resume_path)


# Train!
trainer.fit(model, dataloader)