from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import numpy as np

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine import compile_utils
from tensorflow.python.keras.engine.data_adapter import KerasSequenceAdapter
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CallbackList, ModelCheckpoint #, TensorBoard, ReduceLROnPlateau , LearningRateScheduler

from classifier import build_classifier
from deep_taylor_decomposition import add_deep_taylor_decomposition_to_model_output
from losses import RelevanceGuided
from metrics import SumRelevanceInnerMask, SumRelevance
from sequence import InputSequence

# def learning_rate_scheduler(epoch, lr):
#   return lr * 0.5 if (epoch + 1) % 5 == 0 else lr

def training(training_config, training_data_paths, validation_data_paths, validation_chunk_index, batch_size, accumulate_steps):
  (input_image, input_mask, use_bet, use_rg, input_shape, params_destination, log_destination) = training_config

  K.clear_session()

  # classifier
  (input_tensor, output_tensor) = build_classifier(input_shape + (1,), kernel_count=8, kernel_initializer='he_uniform') 
  
  epochs = 60
  loss_fns = []
  metrics = {}

  if use_rg:
    # relevance-guided extension
    heatmap_tensor = add_deep_taylor_decomposition_to_model_output(output_tensor)
    model = Model(inputs=input_tensor, outputs=[output_tensor, heatmap_tensor])
    
    # losses
    loss_fns.append(tf.keras.losses.CategoricalCrossentropy())
    loss_fns.append(RelevanceGuided())
    
    # metrics
    metrics = {
      'output': [tf.keras.metrics.CategoricalAccuracy()],
      'input_dtd': [SumRelevanceInnerMask(), SumRelevance()],
    }
  else:
    # only classifier
    model = Model(inputs=input_tensor, outputs=output_tensor)

    # losses
    loss_fns.append(tf.keras.losses.CategoricalCrossentropy())

    # metrics
    metrics = {
      'output':[tf.keras.metrics.CategoricalAccuracy()]
    }

  training_sequence = InputSequence(
    training_data_paths,
    [
      input_image,
    ],
    [
      input_mask,
    ],
    input_shape,
    450.,
    batch_size=batch_size,
    use_shuffle=True,
    output_only_categories=not use_rg,
    use_mask_on_input=use_bet,
    excluded_subject_dirnames=[],
  )
  training_data_adapter = KerasSequenceAdapter(
    training_sequence,
    workers=5,
    use_multiprocessing=False,
    max_queue_size=15,
    model=model
  )

  validation_sequence = InputSequence(
    validation_data_paths,
    [
      input_image,
    ],
    [
      input_mask,
    ],
    input_shape,
    450.,
    batch_size=batch_size,
    use_shuffle=False,
    output_only_categories=not use_rg,
    use_mask_on_input=use_bet,
    excluded_subject_dirnames=[],
  )
  validation_data_adapter = KerasSequenceAdapter(
    validation_sequence,
    workers=5,
    use_multiprocessing=False,
    max_queue_size=15,
    model=model
  )

  optimizer = Adam(learning_rate=1e-4)

  callbacks = CallbackList(
    callbacks = [
      ModelCheckpoint(
        params_destination + (
          '/T1_1mm__classifier.val_index_' + str(validation_chunk_index) + '.{epoch:03d}-tcl-{output_loss:.3f}-vcl-{val_output_loss:.3f}-tca-{output_categorical_accuracy:.3f}-vca-{val_output_categorical_accuracy:.3f}-tsrim-{input_dtd_sum_relevance_inner_mask:.3f}-vsrim-{val_input_dtd_sum_relevance_inner_mask:.3f}.h5'
          if use_rg
          else '/T1_1mm__classifier.val_index_' + str(validation_chunk_index) + '.{epoch:03d}-tcl-{loss:.3f}-vcl-{val_loss:.3f}-tca-{categorical_accuracy:.3f}-vca-{val_categorical_accuracy:.3f}.h5'
        ),
        monitor='val_loss',
        save_best_only=False,
        save_weights_only=True,
      ),
      # TensorBoard(
      #   log_dir=os.path.join(log_destination, 'val_index_' + str(validation_chunk_index)),
      #   histogram_freq=0,
      #   write_graph=True,
      #   write_images=False,
      #   update_freq='epoch',
      #   profile_batch=0,
      #   embeddings_freq=0,
      #   embeddings_metadata=None,
      # ),
      # ReduceLROnPlateau(
      #   monitor='val_loss',
      #   factor=0.1,
      #   patience=5,
      #   mode='auto',
      #   min_delta=0.0001,
      #   cooldown=0,
      #   min_lr=0,
      # ),
      # LearningRateScheduler(
      #   learning_rate_scheduler,
      #   verbose=1,
      # ),
    ],
    add_history=True,
    add_progbar=True,
    model=model,
    verbose=1,
    epochs=epochs,
    steps=training_sequence.__len__()
  )
  
  compiled_losses = compile_utils.LossesContainer(
    losses=loss_fns,
    loss_weights=None,
    output_names=model.output_names
  )

  compiled_metrics = compile_utils.MetricsContainer(
    metrics=metrics,
    weighted_metrics=None,
    output_names=model.output_names
  )

  def get_all_metrices():
    return compiled_losses.metrics + compiled_metrics.metrics

  @tf.function
  def train_step(x, y, sw):
    with tf.GradientTape() as tape:
      y_batch_prediction = model(x, training=True)
      loss = compiled_losses(y, y_batch_prediction, sample_weight=sw, regularization_losses=model.losses)
    
    grads = tape.gradient(loss, model.trainable_weights)
    compiled_metrics.update_state(y, y_batch_prediction, sw)

    # Collect metrics to return
    return_metrics = {}
    for metric in get_all_metrices():
      result = metric.result()
      if isinstance(result, dict):
        return_metrics.update(result) 
      else:
        return_metrics[metric.name] = result
    
    return (grads, return_metrics)

  @tf.function
  def test_step(x, y, sw):
    y_batch_prediction = model(x, training=False)
    compiled_losses(y, y_batch_prediction, sample_weight=sw, regularization_losses=model.losses)
    compiled_metrics.update_state(y, y_batch_prediction, sw)

    # Collect metrics to return
    return_metrics = {}
    for metric in get_all_metrices():
      result = metric.result()
      if isinstance(result, dict):
        return_metrics.update(result) 
      else:
        return_metrics[metric.name] = result
    
    return return_metrics

  training_logs = None
  callbacks.on_train_begin()

  for epoch in range(epochs):
    for metric in get_all_metrices():
      metric.reset_states()

    callbacks.on_epoch_begin(epoch)

    # training
    averaged_grads = None
    training_data_iterator = iter(training_data_adapter.get_dataset())
    for step_index in range(training_data_adapter.get_size()):
      callbacks.on_train_batch_begin(step_index)

      batch_train = next(training_data_iterator)
      (x_batch_train, y_batch_train, sample_weight) = tf.keras.utils.unpack_x_y_sample_weight(batch_train)

      (grads, logs) = train_step(x_batch_train, y_batch_train, sample_weight)
      if averaged_grads is None:
        averaged_grads = [g / accumulate_steps for g in grads]
      else:
        for g_index, g in enumerate(grads):
          averaged_grads[g_index] += g / accumulate_steps

      if (step_index + 1) % accumulate_steps == 0 or (step_index + 1) == epochs:
        optimizer.apply_gradients(zip(averaged_grads, model.trainable_weights))
        averaged_grads = None

      callbacks.on_train_batch_end(step_index, logs=logs)

    training_data_adapter.on_epoch_end()
    epoch_logs = copy.copy(logs)

    # validating
    for metric in get_all_metrices():
      metric.reset_states()
    
    callbacks.on_test_begin()

    validation_data_iterator = iter(validation_data_adapter.get_dataset())
    for step_index in range(validation_data_adapter.get_size()):
      callbacks.on_test_batch_begin(step_index)

      batch_validation = next(validation_data_iterator)
      (x_batch_train, y_batch_train, sample_weight) = tf.keras.utils.unpack_x_y_sample_weight(batch_validation)

      val_logs = test_step(x_batch_train, y_batch_train, sample_weight)

      callbacks.on_test_batch_end(step_index, logs=val_logs)

    validation_data_adapter.on_epoch_end()
    callbacks.on_test_end(logs=val_logs)
    val_logs = {'val_' + name: val for name, val in val_logs.items()}
    epoch_logs.update(val_logs)

    callbacks.on_epoch_end(epoch, logs=epoch_logs)
    training_logs = epoch_logs

  callbacks.on_train_end(logs=training_logs)
