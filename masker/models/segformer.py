import torch
from transformers import SegformerConfig, SegformerForSemanticSegmentation
import torch.nn.functional as F

from masker.models.model_wrapper import ModelWrapper


class SegFormerWrapper(ModelWrapper):
    def __init__(self):
        super().__init__(self)
        config = SegformerConfig(
            num_channels=1,
            num_labels=1,
            num_encoder_blocks=4,
            depths=[2, 2, 2, 2],
            sr_ratios=[8, 4, 2, 1],
            hidden_sizes=[32, 64, 160, 256],
            patch_sizes=[7, 3, 3, 3],
            strides=[4, 2, 2, 2],
            num_attention_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            hidden_act="gelu",
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            classifier_dropout_prob=0.1,
            initializer_range=0.02,
            drop_path_rate=0.1,
            layer_norm_eps=1e-6,
            decoder_hidden_size=256,
            semantic_loss_ignore_index=255,
        )

        #config.hidden_sizes = [16, 32, 80, 128]
        #config.decoder_hidden_size = 128

        self.model = SegformerForSemanticSegmentation(config)

    def get_model(self):
        return self.model

    def input_preprocess(self, x):
        return x.view(-1, 1, 1024, 1024)

    def output_postprocess(self, x):
        outputs = x.logits
        outputs = torch.sigmoid(outputs)
        outputs = outputs.view(-1, 256, 256)
        return outputs

    def labels_preprocess(self, x):
        return F.interpolate(x.view(-1, 1, 1024, 1024), size=(256, 256), mode='nearest').view(-1, 256, 256)
