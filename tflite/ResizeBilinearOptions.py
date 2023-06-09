# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class ResizeBilinearOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsResizeBilinearOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = ResizeBilinearOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def ResizeBilinearOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed)

    # ResizeBilinearOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # ResizeBilinearOptions
    def NewHeight(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # ResizeBilinearOptions
    def NewWidth(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

def ResizeBilinearOptionsStart(builder): builder.StartObject(2)
def ResizeBilinearOptionsAddNewHeight(builder, newHeight): builder.PrependInt32Slot(0, newHeight, 0)
def ResizeBilinearOptionsAddNewWidth(builder, newWidth): builder.PrependInt32Slot(1, newWidth, 0)
def ResizeBilinearOptionsEnd(builder): return builder.EndObject()


class ResizeBilinearOptionsT(object):

    # ResizeBilinearOptionsT
    def __init__(self):
        self.newHeight = 0  # type: int
        self.newWidth = 0  # type: int

    @classmethod
    def InitFromBuf(cls, buf, pos):
        resizeBilinearOptions = ResizeBilinearOptions()
        resizeBilinearOptions.Init(buf, pos)
        return cls.InitFromObj(resizeBilinearOptions)

    @classmethod
    def InitFromObj(cls, resizeBilinearOptions):
        x = ResizeBilinearOptionsT()
        x._UnPack(resizeBilinearOptions)
        return x

    # ResizeBilinearOptionsT
    def _UnPack(self, resizeBilinearOptions):
        if resizeBilinearOptions is None:
            return
        self.newHeight = resizeBilinearOptions.NewHeight()
        self.newWidth = resizeBilinearOptions.NewWidth()

    # ResizeBilinearOptionsT
    def Pack(self, builder):
        ResizeBilinearOptionsStart(builder)
        ResizeBilinearOptionsAddNewHeight(builder, self.newHeight)
        ResizeBilinearOptionsAddNewWidth(builder, self.newWidth)
        resizeBilinearOptions = ResizeBilinearOptionsEnd(builder)
        return resizeBilinearOptions
