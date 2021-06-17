from __future__ import absolute_import

import os

from studies import get_studies
from workflow import preprocess

STUDY_BASE_PATH = '/mnt/neuro/nas/STUDIES/'

preprocess(
  get_studies(),
  {
    'T1': STUDY_BASE_PATH + '{study}/MRI_DATA/NII/{subject_id}/data/T1_1mm.M__*.nii.gz',
    'T1_noNeck': STUDY_BASE_PATH + '{study}/MRI_DATA/NII/{subject_id}/AddNii/T1_1mm.noNeck__*.nii.gz',
    'MNI152_1mm_brain_mask': os.path.join(os.path.dirname(os.path.realpath(__file__)), 'MNI152_T1_1mm_brain_mask_dil_ero7.nii.gz'),
  },
  os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '../../data/T1_preprocessed',
  ),
  11
)
