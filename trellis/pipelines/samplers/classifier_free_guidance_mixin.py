from typing import *


class ClassifierFreeGuidanceSamplerMixin:
    """
    A mixin class for samplers that apply classifier-free guidance.
    """

    def _inference_model(self, model, x_t, t, cond, low,neg_cond, cfg_strength, **kwargs):
        if low is None:
            pred = super()._inference_model(model, x_t, t, cond,**kwargs)
            neg_pred = super()._inference_model(model, x_t, t, neg_cond, **kwargs)
            return (1 + cfg_strength) * pred - cfg_strength * neg_pred
        else:
            pred = super()._inference_model(model, x_t, t, cond, low,**kwargs)
            neg_pred = super()._inference_model(model, x_t, t, neg_cond,low, **kwargs)
            return (1 + cfg_strength) * pred - cfg_strength * neg_pred
