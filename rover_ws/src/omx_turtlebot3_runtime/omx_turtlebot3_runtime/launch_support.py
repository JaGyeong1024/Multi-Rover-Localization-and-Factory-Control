"""Launch helpers: runtime Python selection and EEF camera calibration check."""
import math
import os
from pathlib import Path
import sys
import yaml


def runtime_python(prefix, explicit=''):
    if explicit:
        candidates = [Path(explicit).expanduser()]
    elif os.environ.get('OMX_CPP_PYTHON'):
        candidates = [Path(os.environ['OMX_CPP_PYTHON']).expanduser()]
    else:
        prefix = Path(prefix)
        candidates = [base/name/'bin/python' for base in (prefix.parent, prefix.parent.parent)
                      for name in ('.venv_omx_cpp', '.venv_omx_runtime')]
        candidates.append(Path(sys.executable))
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            # Keep the venv launcher path. Resolving its symlink selects system Python.
            return str(candidate.absolute())
    raise ValueError('No executable Python environment; set python_executable or OMX_CPP_PYTHON')


def calibrated_camera(path):
    value = yaml.safe_load(Path(path).read_text())
    if (value.get('image_width'), value.get('image_height')) != (320, 240):
        raise ValueError('Camera calibration must match 320x240')
    k = value.get('camera_matrix', {})
    d = value.get('distortion_coefficients', {})
    matrix, distortion = k.get('data', []), d.get('data', [])
    if (k.get('rows'), k.get('cols')) != (3, 3) or len(matrix) != 9:
        raise ValueError('Invalid camera matrix')
    numbers = matrix+distortion
    if (any(type(x) not in (float, int) or not math.isfinite(x) for x in numbers)
            or matrix[0] <= 0 or matrix[4] <= 0 or matrix[6:] != [0, 0, 1]
            or len(distortion) not in (4, 5, 8, 12, 14)
            or value.get('distortion_model') not in ('plumb_bob', 'rational_polynomial')):
        raise ValueError('Invalid camera intrinsics/distortion')
    return value

