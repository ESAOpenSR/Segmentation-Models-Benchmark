import os
import shutil
from pathlib import Path
import rasterio
import numpy as np

ortho = Path(f'/data/USERS/shollend/combined_download/output/hr_orthofoto/')
gt = Path('/data/USERS/shollend/inferred_buildings/gt')
to_ortho = Path('/data/USERS/shollend/inferred_buildings/base_ortho')

for file in gt.glob('*.tif'):
    name = str(file.name).split('_')[-1]
    img_name = f'HR_ortho_{name}'

    shutil.copy(src=ortho / img_name, dst=to_ortho / img_name)


#
# for img in bil.glob('*.tif'):
#     name = str(img.name).split('_')[1]
#     mask_name = f'HR_mask_{name}'
#     #print(name, mask_name)
#     if mask_name not in files:
#         print('error', mask_name)
#         with rasterio.open(Path("/home/shollend/shares/users/master/dl2/combined_download/output/hr_mask") / mask_name) as src:
#             mask = src.read(1).astype(np.uint8)  # Read first band only
#             mask = np.expand_dims(mask == 41, axis=0)
#
#             with rasterio.open(Path(f'/data/USERS/shollend/inferred_buildings/gt/{mask_name}'), 'w',
#                                **src.profile) as dst:
#                 dst.write(mask)
