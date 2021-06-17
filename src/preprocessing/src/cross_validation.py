from __future__ import absolute_import

import os
import random
import json
import nibabel as nib

from studies import get_studies

this_file_path = os.path.dirname(os.path.realpath(__file__))

study_scans = get_studies()['subject_id']
k = 5
needed_shape = (176, 224, 256)

# subject ids
ASPSF_scans = study_scans['ASPSF']
ProDem_scans = study_scans['ProDem']

# find scans to exclude
excluded_ASPSF_scans = []
for ASPSF_scan in ASPSF_scans:
  image = nib.load(os.path.join(this_file_path, '../../data/input/T1', ASPSF_scan, 'T1_1mm.nii.gz'))
  if image.shape != needed_shape:
    excluded_ASPSF_scans.append(ASPSF_scan)

with open('excluded_ASPSF_scans.json', 'w') as outfile:
  json.dump(excluded_ASPSF_scans, outfile, indent=2)

with open('manually_excluded_ASPSF_scans.json', 'r') as infile:
  manually_excluded_ASPSF_scans = json.load(infile)

excluded_ProDem_scans = []
for ProDem_scan in ProDem_scans:
  image = nib.load(os.path.join(this_file_path, '../../data/input/T1', ProDem_scan, 'T1_1mm.nii.gz'))
  if image.shape != needed_shape:
    excluded_ProDem_scans.append(ProDem_scan)

with open('excluded_ProDem_scans.json', 'w') as outfile:
  json.dump(excluded_ProDem_scans, outfile, indent=2)

with open('manually_excluded_ProDem_scans.json', 'r') as infile:
  manually_excluded_ProDem_scans = json.load(infile)

# exclude scans
ASPSF_scans = set(ASPSF_scans) - set(excluded_ASPSF_scans) - set(manually_excluded_ASPSF_scans)
ASPSF_scans_list = list(ASPSF_scans)
ASPSF_scans_list.sort()
with open('ASPSF_scans.json', 'w') as outfile:
  json.dump(ASPSF_scans_list, outfile, indent=2)

ProDem_scans = set(ProDem_scans) - set(excluded_ProDem_scans) - set(manually_excluded_ProDem_scans)
ProDem_scans_list = list(ProDem_scans)
ProDem_scans_list.sort()
with open('ProDem_scans.json', 'w') as outfile:
  json.dump(ProDem_scans_list, outfile, indent=2)

# distinct subjects
ASPSF_subjects = list(set(map(lambda s: s.split('__')[0], ASPSF_scans)))
ASPSF_subjects.sort()

ProDem_subjects = list(set(map(lambda s: s.split('__')[0], ProDem_scans)))
ProDem_subjects.sort()

# shuffle
random.shuffle(ASPSF_subjects)
random.shuffle(ProDem_subjects)

# calc chunks sizes
ASPSF_chunk_size = round(len(ASPSF_subjects) / k)
if len(ASPSF_subjects) % k != 0:
  ASPSF_chunk_size = ASPSF_chunk_size + 1

ProDem_chunk_size = round(len(ProDem_subjects) / k)
if len(ProDem_subjects) % k != 0:
  ProDem_chunk_size = ProDem_chunk_size + 1

# split
def chunks(lst, n):
  """Yield successive n-sized chunks from lst."""
  for i in range(0, len(lst), n):
    yield lst[i:i + n]

ASPSF_subjects_split = list(chunks(ASPSF_subjects, ASPSF_chunk_size))
x = []
for i in range(len(ASPSF_subjects_split)):
  y = []
  x.append(y)
  for j in range(len(ASPSF_subjects_split[i])):
    for scan in ASPSF_scans:
      if scan.startswith(ASPSF_subjects_split[i][j]):
        y.append(scan)
  print(len(y))

with open('ASPSF_cross_validation_split.json', 'w') as outfile:
  json.dump(x, outfile, indent=2)

ProDem_subjects_split = list(chunks(ProDem_subjects, ProDem_chunk_size))
x = []
for i in range(len(ProDem_subjects_split)):
  y = []
  x.append(y)
  for j in range(len(ProDem_subjects_split[i])):
    for scan in ProDem_scans:
      if scan.startswith(ProDem_subjects_split[i][j]):
        y.append(scan)
  print(len(y))

with open('ProDem_cross_validation_split.json', 'w') as outfile:
  json.dump(x, outfile, indent=2)
