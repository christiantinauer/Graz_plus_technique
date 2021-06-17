from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
from random import shuffle
import nibabel as nib
import numpy as np

from tensorflow.keras.utils import Sequence

class InputSequence(Sequence):
  def __init__(self, data_paths, input_names, mask_names,
               input_shape, normalization_constant,
               batch_size=1, use_shuffle=True, output_only_categories=False, use_mask_on_input=False,
               excluded_subject_dirnames=[]):
    self.data_paths = data_paths
    self.input_names = input_names
    self.mask_names = mask_names
    self.input_shape = input_shape
    self.normalization_constant = normalization_constant
    self.batch_size = batch_size
    self.use_shuffle = use_shuffle
    self.output_only_categories = output_only_categories
    self.use_mask_on_input = use_mask_on_input
    self.filepaths_with_category_and_sample_weighting = []

    for data_path in self.data_paths:
      subject_dirname = data_path.split('/')[-1]
      if subject_dirname in excluded_subject_dirnames:
        continue
      
      category_index = 1 if subject_dirname.startswith('8') else 0
      categorical = np.zeros(2, dtype=np.float32)
      categorical[category_index] = 1.

      self.filepaths_with_category_and_sample_weighting.append((
        [os.path.join(data_path, input_name) for input_name in self.input_names],
        [os.path.join(data_path, mask_name) for mask_name in self.mask_names],
        categorical,
        1.43 if category_index == 1 else 1.
      ))

    if self.use_shuffle:
      shuffle(self.filepaths_with_category_and_sample_weighting)

  def __len__(self):
    return math.ceil(len(self.filepaths_with_category_and_sample_weighting) / self.batch_size)

  def __getitem__(self, index):
    final_batch_size = min(
      self.batch_size,
      len(self.filepaths_with_category_and_sample_weighting) - index * self.batch_size
    )
    batch_filepaths_with_category = self.filepaths_with_category_and_sample_weighting[
      index * self.batch_size:index * self.batch_size + final_batch_size
    ]

    all_input_paths, all_mask_paths, categories, sample_weighting = zip(*batch_filepaths_with_category)

    input_batch = []
    mask_batch = []
    for input_paths_index in range(0, len(all_input_paths)):
      inputs = []
      masks = []

      input_paths = all_input_paths[input_paths_index]
      mask_paths = all_mask_paths[input_paths_index]
      for input_path_index in range(0, len(input_paths)):
        input_path = input_paths[input_path_index]
        mask_path = mask_paths[input_path_index]

        # image
        image = nib.load(input_path)
        if image.shape != self.input_shape:
          print(input_path)
          print(image.shape)
        
        image_data = image.get_fdata(caching='unchanged')
        # image_data = np.nan_to_num(image_data, copy=False, posinf=0., neginf=0.)

        if np.isnan(np.sum(image_data)):
          print(input_path + ' has NaNs.')
        # image_data = np.nan_to_num(image_data, copy=False, posinf=0., neginf=0.)
        
        if np.max(image_data) == 0.:
          print(input_path + ' max is 0.')

        # mask
        mask = nib.load(mask_path)
        if mask.shape != self.input_shape:
          print(mask_path)
          print(mask.shape)
        
        mask_data = mask.get_fdata(caching='unchanged')
        # mask_data = np.nan_to_num(mask_data, copy=False, posinf=0., neginf=0.)
        
        if np.isnan(np.sum(mask_data)):
          print(mask_path + ' has NaNs.')
        # mask_data = np.nan_to_num(mask_data, copy=False, copy=False, posinf=0., neginf=0.)
        
        if np.max(mask_data) == 0.:
          print(mask_path + ' max is 0.')

        if self.use_mask_on_input:
          image_data *= mask_data

        if self.normalization_constant != None:
          image_data = image_data / self.normalization_constant

        inputs.append(image_data)
        masks.append(mask_data)
      
      input_batch.append(inputs)
      mask_batch.append(masks)

    input_batch = np.expand_dims(np.transpose(np.asarray(input_batch), (1, 0, 2, 3, 4)), axis=-1)
    mask_batch = np.expand_dims(np.transpose(np.asarray(mask_batch), (1, 0, 2, 3, 4)), axis=-1)

    input_batch = list(input_batch)
    if len(input_batch) == 1:
      input_batch = input_batch[0]

    mask_batch = list(mask_batch)
    if len(mask_batch) == 1:
      mask_batch = mask_batch[0]

    if self.output_only_categories:
      output_batch = np.asarray(categories)
    else:
      output_batch = [
        np.asarray(categories),
        mask_batch,
      ]

    return (input_batch, output_batch, np.asarray(sample_weighting))

  def on_epoch_end(self):
    if self.use_shuffle:
      shuffle(self.filepaths_with_category_and_sample_weighting)
