import torch

class OptimalTransportFlowMatcher:
    def __init__(self, sigma_min=1e-4):
        """
        Initializes the Flow Matcher for Optimal Transport (OT-CFM).
        
        Args:
            sigma_min: minimum variance.
        """
        self.sigma_min = sigma_min

    def sample_location_and_target(self, x0, x1):
        """
        Generates the intermediate point x_t and the target velocity v_t.
        
        Args:
            x0: noise tensor (Batch, ...), ex: N(0, I)
            x1: real data tensor (Batch, ...)
            
        Returns:
            t: sampled time (Batch, 1)
            x_t: x at instant t
            v_t: target vector field
        """
        batch_size = x0.shape[0]
        
        t = torch.rand((batch_size,), device=x0.device)
        t_reshaped = t.view(batch_size, *[1 for _ in range(x0.ndim - 1)])
        x_t = (1 - t_reshaped) * x0 + t_reshaped * x1
        v_t = x1 - x0
        
        return t, x_t, v_t