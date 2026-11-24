import pytest
import gameplay.pickup_magnet_rollback as mod

CASES = [
    ("swept_attract", (1,)),
    ("object_generation", (1,)),
    ("capacity_first", (1,)),
    ("stable_slot_order", (1,)),
    ("stop_sfx_on_destroy", (1,)),
    ("rollback_restore", (1,)),
    ("attract_step", (1,)),
    ("submit_pickup", (1,)),
    ("grant", (1,)),
    ("reject", (1,)),
    ("validate_capacity", (1,)),
    ("metrics", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
