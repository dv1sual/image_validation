import os
import ctypes
import numpy as np

path_to_dll = "C:/dev/vcpkg/installed/x64-windows/bin"
if path_to_dll not in os.environ["PATH"]:
    os.environ["PATH"] += os.pathsep + path_to_dll

if os.name == "nt":
    _lib = ctypes.CDLL(os.path.join("C:/dev/vcpkg/installed/x64-windows/bin", "OpenImageIO.dll"), ctypes.RTLD_GLOBAL)
else:
    _lib = ctypes.CDLL(os.path.join("C:/dev/vcpkg/installed/x64-windows/bin", "libOpenImageIO.so"), ctypes.RTLD_GLOBAL)


_lib.ImageInput_open.restype = ctypes.c_void_p
_lib.ImageInput_open.argtypes = [ctypes.c_char_p]
_lib.ImageInput_spec.restype = ctypes.POINTER(ctypes.c_void_p)
_lib.ImageInput_spec.argtypes = [ctypes.c_void_p]
_lib.ImageInput_read_image.restype = ctypes.c_bool
_lib.ImageInput_read_image.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
_lib.ImageInput_close.argtypes = [ctypes.c_void_p]


def load_image(image_path):
    input_image = _lib.ImageInput_open(image_path.encode("utf-8"))
    if not input_image:
        raise ValueError(f"Could not open image file: {image_path}")

    spec = _lib.ImageInput_spec(input_image)
    width, height, channels = spec.contents.width, spec.contents.height, spec.contents.nchannels

    image = np.empty((height, width, channels), dtype=np.float32)
    success = _lib.ImageInput_read_image(input_image, image.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    if not success:
        raise ValueError(f"Could not read image file: {image_path}")

    _lib.ImageInput_close(input_image)

    return image
