"""Small FlashAttention-2 API compatibility layer for Pointcept PTv3.

RTX 50-series cards use compute capability 12.0. The external FlashAttention-2
extension does not officially support that architecture, while recent PyTorch
and cuDNN releases do provide fused, memory-efficient SDPA kernels for it.

Pointcept PTv3 only needs ``flash_attn_varlen_qkvpacked_func``. This module
implements that call using fused PyTorch SDPA and deliberately excludes the
quadratic-memory math backend. It is not a general replacement for the full
Dao-AILab flash-attn package.
"""

from .flash_attn_interface import flash_attn_varlen_qkvpacked_func

__all__ = ["flash_attn_varlen_qkvpacked_func"]
__version__ = "2.7.4+ptv3.sdpa.blackwell"
