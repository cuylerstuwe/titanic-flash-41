from ._cute_intrinsics import _asm, dsl_user_op
@dsl_user_op
def fence_acq_rel_sys(*, loc=None, ip=None):
    _asm(None, [], "fence.acq_rel.sys;", "", loc=loc, ip=ip)
