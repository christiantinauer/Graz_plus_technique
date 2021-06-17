import os

from nipype.interfaces.utility import IdentityInterface, Function, Select
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.pipeline.engine import Workflow, Node
from nipype.interfaces.fsl import BET, FLIRT, ApplyXFM, FNIRT, ApplyWarp

# get container
def datasink_container_from_identity(study, subject_id):
  return study + '/' + subject_id

def build_selectfiles_workflow(studies, select_templates, name='selectfiles'):
  # identities
  study_identity_node = Node(IdentityInterface(
    fields=['study'],
  ), name=name + '__study_identity')
  study_identity_node.iterables = [
    ('study', studies['study']),
  ]

  subject_identity_node = Node(IdentityInterface(
    fields=['study', 'subject_id'],
  ), name=name + '__subject_identity')
  subject_identity_node.itersource = (name + '__study_identity', 'study')
  subject_identity_node.iterables = [('subject_id', studies['subject_id'])]
  
  # select files
  selectfiles_node = Node(SelectFiles(
    select_templates,
    sort_filelist=True,
    force_lists=['T1', 'T1_noNeck'],
  ), name=name + '__selectfiles')

  # datasink container
  datasink_container_node = Node(Function(
    input_names=['study', 'subject_id'],
    output_names=['container'],
    function=datasink_container_from_identity,
  ), name=name + '__datasink_container')

  # select files subworkflow
  selectfiles_worklow = Workflow(name)
  selectfiles_worklow.connect([
    (study_identity_node, subject_identity_node,        [('study', 'study')]),
    (subject_identity_node, selectfiles_node,           [
                                                          ('study', 'study'),
                                                          ('subject_id', 'subject_id'),
                                                        ]),
    (subject_identity_node, datasink_container_node,    [
                                                          ('study', 'study'),
                                                          ('subject_id', 'subject_id'),
                                                        ]),
  ])

  return selectfiles_worklow

def build_datasink_node(target_base_dir, workflow_name):
  return Node(DataSink(
    base_directory=target_base_dir,
    parameterization=False,
    substitutions=[
      ('T1_1mm_brain_mask_nlin', ''),
    ],
    regexp_substitutions=[
      (r'T1_1mm.M__[0-9]+.nii.gz', 'T1_1mm.nii.gz'),
      (r'T1_1mm_brain_mask/T1_1mm.noNeck__[0-9]+_brain_mask.nii.gz', 'T1_1mm__brain_mask.nii.gz'),
      (r'T1_1mm_brain_mask_dof6/T1_1mm.noNeck__[0-9]+_brain_mask_flirt.nii.gz', 'T1_1mm__brain_mask_@_MNI152_1mm_dof6.nii.gz'),
      (r'regT1_brain_mask_nlin/T1_1mm.noNeck__[0-9]+_brain_mask_warp.nii.gz', 'T1_1mm__brain_mask_@_MNI152_1mm_nlin.nii.gz'),

      (r'regMatrix_dof6/T1_1mm.noNeck__[0-9]+_brain_flirt.mat', 'T1_1mm_to_MNI152_1mm_dof6.mat'),
      (r'regMatrix_dof12/T1_1mm.noNeck__[0-9]+_brain_flirt.mat', 'T1_1mm_to_MNI152_1mm_dof12.mat'),
      (r'regFieldFile_nlin/T1_1mm.M__[0-9]+_field.nii.gz', 'T1_1mm_to_MNI152_warp.nii.gz'),
      
      (r'regT1_dof6/T1_1mm.M__[0-9]+_flirt.nii.gz', 'T1_1mm_@_MNI152_1mm_dof6.nii.gz'),
      (r'regT1_dof12/T1_1mm.M__[0-9]+_flirt.nii.gz', 'T1_1mm_@_MNI152_1mm_dof12.nii.gz'),
      (r'regT1_nlin/T1_1mm.M__[0-9]+_warp.nii.gz', 'T1_1mm_@_MNI152_1mm_nlin.nii.gz'),
    ]
  ), name=workflow_name + '__datasink')

def preprocess(studies, select_templates, target_base_dir, processor_count=1):
  # preprocess workflow
  preprocess_workflow = Workflow(
    name='preprocess',
    base_dir=os.path.join(target_base_dir, 'tmp')
  )

  # selectfiles subworkflow
  selectfiles_workflow = build_selectfiles_workflow(
    studies, select_templates,
  )

  # datasink
  datasink_node = build_datasink_node(target_base_dir, 'T1')

  # select first T1
  select_first_T1_node = Node(Select(
    index=[0],
  ), name='T1__select_first_T1')

  # select first T1_noNeck
  select_first_T1_noNeck_node = Node(Select(
    index=[0],
  ), name='T1__select_first_T1_noNeck')

  # bet <input> <output> -d -s -m -f 0.35 -B
  bet_node = Node(BET(
    frac=0.35,
    mask=True,
    reduce_bias=True,
  ), name='T1__bet')

  # flirt -cost corratio -dof 6 -in <input> -ref <reference> -out <output> -omat <outputmatrix>
  reg_to_MNI_dof6_node = Node(FLIRT(
    cost='corratio',
    dof=6,
    reference='/opt/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz',
  ), name='T1__reg_to_MNI_dof6')

  # flirt -applyxfm -in <input> -ref <reference> -init <dof6_matrix> -out <output>
  apply_to_MNI_dof6_node = Node(ApplyXFM(
    apply_xfm=True,
    reference='/opt/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz',
  ), name='T1__apply_to_MNI_dof6')

  # flirt -applyxfm -in <input> -ref <reference> -init <dof6_matrix> -out <output>
  apply_brain_mask_to_MNI_dof6_node = Node(ApplyXFM(
    apply_xfm=True,
    interp='nearestneighbour',
    reference='/opt/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz',
  ), name='T1__apply_brain_mask_to_MNI_dof6')

  # flirt -cost corratio -dof 12 -in <input> -ref <reference> -init <dof6_matrix> -out <output> -omat <outputmatrix>
  reg_to_MNI_dof12_node = Node(FLIRT(
    cost='corratio',
    dof=12,
    reference='/opt/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz',
  ), name='T1__reg_to_MNI_dof12')

  # flirt -applyxfm -in <input> -ref <reference> -init <dof12_matrix> -out <output>
  apply_to_MNI_dof12_node = Node(ApplyXFM(
    apply_xfm=True,
    reference='/opt/fsl/data/standard/MNI152_T1_1mm.nii.gz',
  ), name='T1__apply_to_MNI_dof12')

  # fnirt --in=<input> --ref=<reference> --config=T1_2_MNI152_2mm --aff=<dof12_matrix> --iout=<output> --fout=<field_output>
  reg_to_MNI_nlin_node = Node(FNIRT(
    config_file='T1_2_MNI152_2mm',
    field_file=True,
    ref_file='/opt/fsl/data/standard/MNI152_T1_2mm.nii.gz',
  ), name='T1__reg_to_MNI_nlin')

  # applywarp -i <input> -o <output> -r <reference> -w <field_output>
  apply_to_MNI_nlin_node = Node(ApplyWarp(
    ref_file='/opt/fsl/data/standard/MNI152_T1_1mm.nii.gz',
  ), name='T1__apply_to_MNI_nlin')

  # applywarp -i <input> -o <output> -r <reference> -w <field_output>
  apply_brain_mask_to_MNI_nlin_node = Node(ApplyWarp(
    ref_file='/opt/fsl/data/standard/MNI152_T1_1mm.nii.gz',
    interp='nn',
  ), name='T1__apply_brain_mask_to_MNI_nlin')

  # T1
  preprocess_workflow.connect([
    (selectfiles_workflow, select_first_T1_node,                    [('selectfiles__selectfiles.T1', 'inlist')]),
    (selectfiles_workflow, select_first_T1_noNeck_node,             [('selectfiles__selectfiles.T1_noNeck', 'inlist')]),
    
    (select_first_T1_noNeck_node, bet_node,                         [('out', 'in_file')]),
    
    (bet_node, reg_to_MNI_dof6_node,                                [('out_file', 'in_file')]),
    (bet_node, reg_to_MNI_dof12_node,                               [('out_file', 'in_file')]),

    (select_first_T1_node, apply_to_MNI_dof6_node,                  [('out', 'in_file')]),
    (reg_to_MNI_dof6_node, apply_to_MNI_dof6_node,                  [('out_matrix_file', 'in_matrix_file')]),

    (bet_node, apply_brain_mask_to_MNI_dof6_node,                   [('mask_file', 'in_file')]),
    (reg_to_MNI_dof6_node, apply_brain_mask_to_MNI_dof6_node,       [('out_matrix_file', 'in_matrix_file')]),

    (select_first_T1_node, apply_to_MNI_dof12_node,                 [('out', 'in_file')]),
    (reg_to_MNI_dof12_node, apply_to_MNI_dof12_node,                [('out_matrix_file', 'in_matrix_file')]),

    (select_first_T1_node, reg_to_MNI_nlin_node,                    [('out', 'in_file')]),
    (reg_to_MNI_dof12_node, reg_to_MNI_nlin_node,                   [('out_matrix_file', 'affine_file')]),

    (select_first_T1_node, apply_to_MNI_nlin_node,                  [('out', 'in_file')]),
    (reg_to_MNI_nlin_node, apply_to_MNI_nlin_node,                  [('field_file', 'field_file')]),

    (bet_node, apply_brain_mask_to_MNI_nlin_node,                   [('mask_file', 'in_file')]),
    (reg_to_MNI_nlin_node, apply_brain_mask_to_MNI_nlin_node,       [('field_file', 'field_file')]),

    (selectfiles_workflow, datasink_node,                           [
      ('selectfiles__datasink_container.container', 'container'),
      ('selectfiles__selectfiles.MNI152_1mm_brain_mask', 'T1_1mm_brain_mask_nlin'),
                                                                    ]),
    (select_first_T1_node, datasink_node,                           [('out', '@T1_1mm')]),
    (bet_node, datasink_node,                                       [('mask_file', 'T1_1mm_brain_mask')]),
    (apply_brain_mask_to_MNI_dof6_node, datasink_node,              [('out_file', 'T1_1mm_brain_mask_dof6')]),
    (reg_to_MNI_dof6_node, datasink_node,                           [('out_matrix_file', 'regMatrix_dof6')]),
    (reg_to_MNI_dof12_node, datasink_node,                          [('out_matrix_file', 'regMatrix_dof12')]),
    (apply_to_MNI_dof6_node, datasink_node,                         [('out_file', 'regT1_dof6')]),
    (apply_to_MNI_dof12_node, datasink_node,                        [('out_file', 'regT1_dof12')]),
    (reg_to_MNI_nlin_node, datasink_node,                           [('field_file', 'regFieldFile_nlin')]),
    (apply_to_MNI_nlin_node, datasink_node,                         [('out_file', 'regT1_nlin')]),
    (apply_brain_mask_to_MNI_nlin_node, datasink_node,              [('out_file', 'regT1_brain_mask_nlin')]),
  ])

  preprocess_workflow.write_graph(dotfilename='./graphs/preprocess.dot', graph2use='orig', simple_form=True)
  preprocess_workflow.run('MultiProc', plugin_args={'n_procs': processor_count})
