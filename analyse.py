#!/usr/bin/env python

import flatbuffers
import numpy as np
import sys
sys.path.append("tflite/")
import Model

def load_model_from_file(model_filename):
  with open(model_filename, "rb") as file:
    buffer_data = file.read()
  model_obj = Model.Model.GetRootAsModel(buffer_data, 0)
  model = Model.ModelT.InitFromObj(model_obj)
  return model

def save_model_to_file(model, model_filename):
  builder = flatbuffers.Builder(1024)
  model_offset = model.Pack(builder)
  builder.Finish(model_offset, file_identifier=b'TFL3')
  model_data = builder.Output()
  with open(model_filename, 'wb') as out_file:
    out_file.write(model_data)

model_filename = sys.argv[1]

model = load_model_from_file(model_filename)

for subgraph in model.subgraphs:
  tensors = subgraph.tensors
  operators = subgraph.operators

  tensor_first_write = [-1] * len(tensors)
  for input in subgraph.inputs:
    tensor_first_write[input] = 0
  for step, operator in enumerate(operators):
    for output in operator.outputs:
      if tensor_first_write[output] == -1:
        tensor_first_write[output] = step

  tensor_last_read = [-1] * len(tensors)
  for output in subgraph.outputs:
    tensor_last_read[output] = len(operators) - 1
  for step, operator in reversed(list(enumerate(operators))):
    for input in operator.inputs:
      if tensor_last_read[input] == -1:
        tensor_last_read[input] = step

  print(tensor_first_write)
  print(tensor_last_read)

  for step, operator in enumerate(operators):
    total_memory = 0
    for tensor_index, tensor in enumerate(tensors):
      first_write = tensor_first_write[tensor_index]
      last_read = tensor_last_read[tensor_index]
      if first_write == -1 or last_read == -1:
        continue
      if step < first_write or step > last_read:
        continue
      shape = tensor.shape
      element_count = 1
      for dim in shape:
        element_count *= dim
      total_memory += element_count
    print("%d: %s elements" % (step, f'{total_memory:,}'))
