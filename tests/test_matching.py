import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from src.matching import OptimalTransportFlowMatcher

def test_ot_flow_matching_shapes_2d():
    matcher = OptimalTransportFlowMatcher()
    batch_size = 32
    dim = 2
    
    x0 = torch.randn(batch_size, dim)
    x1 = torch.randn(batch_size, dim)
    
    t, x_t, v_t = matcher.sample_location_and_target(x0, x1)
    
    assert t.shape == (batch_size,)
    assert x_t.shape == (batch_size, dim)
    assert v_t.shape == (batch_size, dim)

def test_ot_flow_matching_shapes_images():
    matcher = OptimalTransportFlowMatcher()
    batch_size = 8
    channels = 1
    height = 28
    width = 28
    
    x0 = torch.randn(batch_size, channels, height, width)
    x1 = torch.randn(batch_size, channels, height, width)
    
    t, x_t, v_t = matcher.sample_location_and_target(x0, x1)
    
    assert t.shape == (batch_size,)
    assert x_t.shape == (batch_size, channels, height, width)
    assert v_t.shape == (batch_size, channels, height, width)

def test_ot_flow_matching_interpolation():
    matcher = OptimalTransportFlowMatcher()
    
    # Use explicit tensors to easily verify the math
    x0 = torch.tensor([[0.0, 0.0]])
    x1 = torch.tensor([[10.0, 10.0]])
    
    _, _, v_t = matcher.sample_location_and_target(x0, x1)
    
    expected_vt = torch.tensor([[10.0, 10.0]])
    assert torch.allclose(v_t, expected_vt)