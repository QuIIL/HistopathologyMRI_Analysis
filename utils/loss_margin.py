import numpy as np


class LossMargin:
    """Compute the decayed loss margin"""
    def __init__(self, total_iter, margin_decay_rate=3, initial_margin=1e-1):
        decay_period_fraction = 1
        self.decay_length = int(total_iter * decay_period_fraction)
        _step = np.linspace(0, 1, self.decay_length)
        _factor = 1 - _step ** margin_decay_rate if margin_decay_rate > 0 else 1 - _step * 0

        self.margins = initial_margin * _factor

    def get_margin(self, iteration):
        if iteration >= self.decay_length:
            return 0
        return self.margins[iteration]
