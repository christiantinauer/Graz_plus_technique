from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json

import tensorflow as tf
import numpy as np

# Set seed for experiment reproducibility
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

from training import training

this_file_path = os.path.dirname(os.path.realpath(__file__))
data_base_path = os.path.join(this_file_path, '../../data/input/T1')

with open('ASPSF_cross_validation_split.json', 'r') as infile:
  ASPSF_scans_split = json.load(infile)

with open('ProDem_cross_validation_split.json', 'r') as infile:
  ProDem_scans_split = json.load(infile)

for training_config in [
  # image                               mask                                            bet     rg      shape             params destination                        logs destination  
  ('T1_1mm.nii.gz',                     'T1_1mm__brain_mask.nii.gz',                    False,  True,   (176, 224, 256),  '../../weights/T1_RG',                    '../../logs/T1_RG'),
  ('T1_1mm_@_MNI152_1mm_dof6.nii.gz',   'T1_1mm__brain_mask_@_MNI152_1mm_dof6.nii.gz',  False,  True,   (182, 218, 182),  '../../weights/T1@MNI152_dof6_RG',        '../../logs/T1@MNI152_dof6_RG'),
  ('T1_1mm_@_MNI152_1mm_nlin.nii.gz',   'T1_1mm__brain_mask_@_MNI152_1mm_nlin.nii.gz',  False,  True,   (182, 218, 182),  '../../weights/T1@MNI152_nlin_RG',        '../../logs/T1@MNI152_nlin_RG'),
]:
  for validation_chunk_index in range(len(ASPSF_scans_split)):
    validation_data_paths = [os.path.join(data_base_path, scan_dir) for scan_dir in ASPSF_scans_split[validation_chunk_index] + ProDem_scans_split[validation_chunk_index]]

    training_data_paths = []
    for training_chunk_index in range(len(ASPSF_scans_split)):
      if training_chunk_index == validation_chunk_index:
        continue
      
      training_data_paths = training_data_paths + [os.path.join(data_base_path, scan_dir) for scan_dir in ASPSF_scans_split[training_chunk_index] + ProDem_scans_split[training_chunk_index]]

    # selection is done with visible device...
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # ... and not tf.device because it fails when first card is already in use
    with tf.device('/device:GPU:0'):
      training(training_config, training_data_paths, validation_data_paths, validation_chunk_index, 2, 4)
