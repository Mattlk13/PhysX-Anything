from typing import *


class GuidanceIntervalSamplerMixin:
    """
    A mixin class for samplers that apply classifier-free guidance with interval.
    """

    def _inference_model(self, model, x_t, t, cond, low, neg_cond, cfg_strength, cfg_interval, **kwargs):
        if low is None:
            if cfg_interval[0] <= t <= cfg_interval[1]:
                pred = super()._inference_model(model, x_t, t, cond, **kwargs)
                neg_pred = super()._inference_model(model, x_t, t, neg_cond, **kwargs)
                return (1 + cfg_strength) * pred - cfg_strength * neg_pred
            else:
                return super()._inference_model(model, x_t, t, cond, **kwargs)
        else:
            if cfg_interval[0] <= t <= cfg_interval[1]:
                pred = super()._inference_model(model, x_t, t, cond, low, **kwargs)
                neg_pred = super()._inference_model(model, x_t, t, neg_cond, low, **kwargs)
                return (1 + cfg_strength) * pred - cfg_strength * neg_pred
            else:
                return super()._inference_model(model, x_t, t, cond, low, **kwargs)
            
class GuidanceIntervalSamplerMixin_ori:
    """
    A mixin class for samplers that apply classifier-free guidance with interval.
    """

    def _inference_model(self, model, x_t, t, cond, neg_cond, cfg_strength, cfg_interval, **kwargs):

            if cfg_interval[0] <= t <= cfg_interval[1]:
                pred = super()._inference_model(model, x_t, t, cond, **kwargs)
                neg_pred = super()._inference_model(model, x_t, t, neg_cond, **kwargs)
                return (1 + cfg_strength) * pred - cfg_strength * neg_pred
            else:
                return super()._inference_model(model, x_t, t, cond, **kwargs)
        