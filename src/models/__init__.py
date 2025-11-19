# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2025 Fabio Oddi
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
#  You may use, modify, and redistribute this file according to the terms of the
#  license. Attribution is required if this code is used in other works.
# ------------------------------------------------------------------------------

"""Built-in models and plugin registrations."""

# Import movement, detection, and logic packages so their registration side effects run.
from . import movement  # noqa: F401
from . import detection  # noqa: F401
from . import logic  # noqa: F401
from . import motion  # noqa: F401
