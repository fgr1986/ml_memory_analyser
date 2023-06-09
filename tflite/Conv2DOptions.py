# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class Conv2DOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsConv2DOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Conv2DOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def Conv2DOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed)

    # Conv2DOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # Conv2DOptions
    def Padding(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # Conv2DOptions
    def StrideW(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # Conv2DOptions
    def StrideH(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # Conv2DOptions
    def FusedActivationFunction(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

def Conv2DOptionsStart(builder): builder.StartObject(4)
def Conv2DOptionsAddPadding(builder, padding): builder.PrependInt8Slot(0, padding, 0)
def Conv2DOptionsAddStrideW(builder, strideW): builder.PrependInt32Slot(1, strideW, 0)
def Conv2DOptionsAddStrideH(builder, strideH): builder.PrependInt32Slot(2, strideH, 0)
def Conv2DOptionsAddFusedActivationFunction(builder, fusedActivationFunction): builder.PrependInt8Slot(3, fusedActivationFunction, 0)
def Conv2DOptionsEnd(builder): return builder.EndObject()


class Conv2DOptionsT(object):

    # Conv2DOptionsT
    def __init__(self):
        self.padding = 0  # type: int
        self.strideW = 0  # type: int
        self.strideH = 0  # type: int
        self.fusedActivationFunction = 0  # type: int

    @classmethod
    def InitFromBuf(cls, buf, pos):
        conv2DOptions = Conv2DOptions()
        conv2DOptions.Init(buf, pos)
        return cls.InitFromObj(conv2DOptions)

    @classmethod
    def InitFromObj(cls, conv2DOptions):
        x = Conv2DOptionsT()
        x._UnPack(conv2DOptions)
        return x

    # Conv2DOptionsT
    def _UnPack(self, conv2DOptions):
        if conv2DOptions is None:
            return
        self.padding = conv2DOptions.Padding()
        self.strideW = conv2DOptions.StrideW()
        self.strideH = conv2DOptions.StrideH()
        self.fusedActivationFunction = conv2DOptions.FusedActivationFunction()

    # Conv2DOptionsT
    def Pack(self, builder):
        Conv2DOptionsStart(builder)
        Conv2DOptionsAddPadding(builder, self.padding)
        Conv2DOptionsAddStrideW(builder, self.strideW)
        Conv2DOptionsAddStrideH(builder, self.strideH)
        Conv2DOptionsAddFusedActivationFunction(builder, self.fusedActivationFunction)
        conv2DOptions = Conv2DOptionsEnd(builder)
        return conv2DOptions
