"""Package init for neuronpedia_inference.

This runs before any submodule is imported, so it's the right place to
apply process-wide configuration that other modules (and their imported
nnsight code) must observe at import time.
"""

# IMPORTANT: Disable nnsight's PYMOUNT before any nnsight tracer can be
# created elsewhere in the package.
#
# nnsight v0.5.x uses a process-wide stack counter to mount/unmount
# `.save` and `.stop` on the C-level PyBaseObject dict. The unmount path
# has a bug: `PyDict_DelItemString` failures aren't propagated, leading
# to `SystemError: <built-in function unmount> returned a result with an
# exception set`. Once that fires, the C-level mount registry stays
# corrupted for the lifetime of the process and every subsequent tracer
# request fails. We never call `.save()` directly (we use
# `nnsight.save(...)` instead), so disabling PYMOUNT is safe and
# eliminates the buggy code path entirely.
import nnsight as _nnsight

_nnsight.CONFIG.APP.PYMOUNT = False
