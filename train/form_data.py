import os
import shutil

# directory = "/workspace/ControlNet/image_log/train/myLog/" + "gs-000500_e-" + str(61).zfill(6)+"/samples_cfg_scale_9.00_gs-166288_e-000062_b-000500.png"
# os.rename("/workspace/ControlNet/image_log/train/myLog/samples_cfg_scale_9.00_gs-166288_e-000062_b-000500.png", directory)
path = "image_log/train"
dir_list = os.listdir(path)
for files in dir_list:
    
    if not files.endswith('.png'): continue
    
    if files.startswith('control_'): dir_name = path + files.replace('control_', '/')
    if files.startswith('reconstruction_'): dir_name = path + files.replace('reconstruction_', '/')
    if files.startswith('samples_cfg_scale_9.00_'): dir_name = path + files.replace('samples_cfg_scale_9.00_', '/')
    if files.startswith('conditioning_gs_'): dir_name = path + files.replace('conditioning_gs_', '/')
    dir_name = dir_name[:-4]
    
    source_path = path + '/' + files
    dest_path = dir_name + '/' + files
    
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        
    shutil.move(source_path, dest_path)