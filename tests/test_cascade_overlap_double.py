import pytest
import render.cascade_overlap_double as mod

CASES = [
    ("split_no_overlap", (4, 2)),
    ("bias_by_texel", (1024, 2)),
    ("single_layer_sample", (1, 2)),
    ("contact_shadow", (1, 2)),
    ("cascade_stabilize", (1, 2)),
    ("texel_size", (1.0, 2.0)),
    ("overlap_resolve", (1, 2)),
    ("frustum_sphere", (1, 2)),
    ("blend_region", (1, 2)),
    ("depth_compare", (1, 2)),
    ("pcf_radius", (2, 3)),
    ("resolution_switch", (1, 2)),
    ("acne_guard", (0.5, 0.2)),
    ("validation", (1, 2)),
    ("double_edge_metric", (1.0, 2.0)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
