# -*- coding: UTF-8 -*-
"""
ftsc
~~~~~~~~~~~~
Fast Time Series clustering using approximated distance matrices
:author: Mathias Pede
:license: Apache License, Version 2.0, see LICENSE for details.
"""
import logging

logger = logging.getLogger("ftsc")

from . import aca, solradm, data_loader, clustering, cluster_problem

try:
    from . import ed_c
except ImportError:
    ed_c = None
try:
    from . import msm_c
except ImportError:
    msm_c = None
try:
    from . import triangle_fixing_c
except ImportError:
    triangle_fixing_c = None
