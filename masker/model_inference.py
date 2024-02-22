import deepspeed
import torch

from masker.models.model_factory import create_model

class ModelInference:
    def __init__(self, model_name):
        self.model = create_model()

        self.engine = deepspeed.init_inference(
            model=self.model,
            checkpoint=model_name,
            dtype=torch.float32
        )

    def do_inference(self, input_tensor):
        return self.engine.module(input_tensor.cuda()).cpu()
