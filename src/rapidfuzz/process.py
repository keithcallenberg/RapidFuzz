# SPDX-License-Identifier: MIT
# Copyright (C) 2021 Max Bachmann

from rapidfuzz.cpp_process import extract, extractOne, extract_iter

try:
    from rapidfuzz.cpp_process_cdist import cdist
except ImportError:
    def cdist(*args, **kwargs):
        raise NotImplementedError("implementation requires numpy to be installed")

try:
    from rapidfuzz.cpp_process_eofm import eofm
except ImportError:
    def eofm(*args, **kwargs):
        raise NotImplementedError("implementation requires numpy to be installed")
