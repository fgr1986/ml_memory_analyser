# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class Operator(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsOperator(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Operator()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def OperatorBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed)

    # Operator
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # Operator
    def OpcodeIndex(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint32Flags, o + self._tab.Pos)
        return 0

    # Operator
    def Inputs(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Int32Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return 0

    # Operator
    def InputsAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int32Flags, o)
        return 0

    # Operator
    def InputsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Operator
    def InputsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        return o == 0

    # Operator
    def Outputs(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Int32Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return 0

    # Operator
    def OutputsAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int32Flags, o)
        return 0

    # Operator
    def OutputsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Operator
    def OutputsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        return o == 0

    # Operator
    def BuiltinOptionsType(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint8Flags, o + self._tab.Pos)
        return 0

    # Operator
    def BuiltinOptions(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            from flatbuffers.table import Table
            obj = Table(bytearray(), 0)
            self._tab.Union(obj, o)
            return obj
        return None

    # Operator
    def CustomOptions(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Uint8Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 1))
        return 0

    # Operator
    def CustomOptionsAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Uint8Flags, o)
        return 0

    # Operator
    def CustomOptionsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Operator
    def CustomOptionsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        return o == 0

def OperatorStart(builder): builder.StartObject(6)
def OperatorAddOpcodeIndex(builder, opcodeIndex): builder.PrependUint32Slot(0, opcodeIndex, 0)
def OperatorAddInputs(builder, inputs): builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(inputs), 0)
def OperatorStartInputsVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def OperatorAddOutputs(builder, outputs): builder.PrependUOffsetTRelativeSlot(2, flatbuffers.number_types.UOffsetTFlags.py_type(outputs), 0)
def OperatorStartOutputsVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def OperatorAddBuiltinOptionsType(builder, builtinOptionsType): builder.PrependUint8Slot(3, builtinOptionsType, 0)
def OperatorAddBuiltinOptions(builder, builtinOptions): builder.PrependUOffsetTRelativeSlot(4, flatbuffers.number_types.UOffsetTFlags.py_type(builtinOptions), 0)
def OperatorAddCustomOptions(builder, customOptions): builder.PrependUOffsetTRelativeSlot(5, flatbuffers.number_types.UOffsetTFlags.py_type(customOptions), 0)
def OperatorStartCustomOptionsVector(builder, numElems): return builder.StartVector(1, numElems, 1)
def OperatorEnd(builder): return builder.EndObject()

import tflite.AddOptions
import tflite.BuiltinOptions
import tflite.CallOptions
import tflite.ConcatEmbeddingsOptions
import tflite.ConcatenationOptions
import tflite.Conv2DOptions
import tflite.DepthwiseConv2DOptions
import tflite.FullyConnectedOptions
import tflite.L2NormOptions
import tflite.LSHProjectionOptions
import tflite.LSTMOptions
import tflite.LocalResponseNormalizationOptions
import tflite.Pool2DOptions
import tflite.RNNOptions
import tflite.ReshapeOptions
import tflite.ResizeBilinearOptions
import tflite.SVDFOptions
import tflite.SkipGramOptions
import tflite.SoftmaxOptions
import tflite.SpaceToDepthOptions
try:
    from typing import List, Union
except:
    pass

class OperatorT(object):

    # OperatorT
    def __init__(self):
        self.opcodeIndex = 0  # type: int
        self.inputs = None  # type: List[int]
        self.outputs = None  # type: List[int]
        self.builtinOptionsType = 0  # type: int
        self.builtinOptions = None  # type: Union[None, tflite.Conv2DOptions.Conv2DOptionsT, tflite.DepthwiseConv2DOptions.DepthwiseConv2DOptionsT, tflite.ConcatEmbeddingsOptions.ConcatEmbeddingsOptionsT, tflite.LSHProjectionOptions.LSHProjectionOptionsT, tflite.Pool2DOptions.Pool2DOptionsT, tflite.SVDFOptions.SVDFOptionsT, tflite.RNNOptions.RNNOptionsT, tflite.FullyConnectedOptions.FullyConnectedOptionsT, tflite.SoftmaxOptions.SoftmaxOptionsT, tflite.ConcatenationOptions.ConcatenationOptionsT, tflite.AddOptions.AddOptionsT, tflite.L2NormOptions.L2NormOptionsT, tflite.LocalResponseNormalizationOptions.LocalResponseNormalizationOptionsT, tflite.LSTMOptions.LSTMOptionsT, tflite.ResizeBilinearOptions.ResizeBilinearOptionsT, tflite.CallOptions.CallOptionsT, tflite.ReshapeOptions.ReshapeOptionsT, tflite.SkipGramOptions.SkipGramOptionsT, tflite.SpaceToDepthOptions.SpaceToDepthOptionsT]
        self.customOptions = None  # type: List[int]

    @classmethod
    def InitFromBuf(cls, buf, pos):
        operator = Operator()
        operator.Init(buf, pos)
        return cls.InitFromObj(operator)

    @classmethod
    def InitFromObj(cls, operator):
        x = OperatorT()
        x._UnPack(operator)
        return x

    # OperatorT
    def _UnPack(self, operator):
        if operator is None:
            return
        self.opcodeIndex = operator.OpcodeIndex()
        if not operator.InputsIsNone():
            if np is None:
                self.inputs = []
                for i in range(operator.InputsLength()):
                    self.inputs.append(operator.Inputs(i))
            else:
                self.inputs = operator.InputsAsNumpy()
        if not operator.OutputsIsNone():
            if np is None:
                self.outputs = []
                for i in range(operator.OutputsLength()):
                    self.outputs.append(operator.Outputs(i))
            else:
                self.outputs = operator.OutputsAsNumpy()
        self.builtinOptionsType = operator.BuiltinOptionsType()
        self.builtinOptions = tflite.BuiltinOptions.BuiltinOptionsCreator(self.builtinOptionsType, operator.BuiltinOptions())
        if not operator.CustomOptionsIsNone():
            if np is None:
                self.customOptions = []
                for i in range(operator.CustomOptionsLength()):
                    self.customOptions.append(operator.CustomOptions(i))
            else:
                self.customOptions = operator.CustomOptionsAsNumpy()

    # OperatorT
    def Pack(self, builder):
        if self.inputs is not None:
            if np is not None and type(self.inputs) is np.ndarray:
                inputs = builder.CreateNumpyVector(self.inputs)
            else:
                OperatorStartInputsVector(builder, len(self.inputs))
                for i in reversed(range(len(self.inputs))):
                    builder.PrependInt32(self.inputs[i])
                inputs = builder.EndVector(len(self.inputs))
        if self.outputs is not None:
            if np is not None and type(self.outputs) is np.ndarray:
                outputs = builder.CreateNumpyVector(self.outputs)
            else:
                OperatorStartOutputsVector(builder, len(self.outputs))
                for i in reversed(range(len(self.outputs))):
                    builder.PrependInt32(self.outputs[i])
                outputs = builder.EndVector(len(self.outputs))
        if self.builtinOptions is not None:
            builtinOptions = self.builtinOptions.Pack(builder)
        if self.customOptions is not None:
            if np is not None and type(self.customOptions) is np.ndarray:
                customOptions = builder.CreateNumpyVector(self.customOptions)
            else:
                OperatorStartCustomOptionsVector(builder, len(self.customOptions))
                for i in reversed(range(len(self.customOptions))):
                    builder.PrependUint8(self.customOptions[i])
                customOptions = builder.EndVector(len(self.customOptions))
        OperatorStart(builder)
        OperatorAddOpcodeIndex(builder, self.opcodeIndex)
        if self.inputs is not None:
            OperatorAddInputs(builder, inputs)
        if self.outputs is not None:
            OperatorAddOutputs(builder, outputs)
        OperatorAddBuiltinOptionsType(builder, self.builtinOptionsType)
        if self.builtinOptions is not None:
            OperatorAddBuiltinOptions(builder, builtinOptions)
        if self.customOptions is not None:
            OperatorAddCustomOptions(builder, customOptions)
        operator = OperatorEnd(builder)
        return operator