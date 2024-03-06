import torch

from masker.trainer_config import create_model, create_loss


class ModelInference:
    def __init__(self, model_name):
        self.model_wrapper = create_model()
        self.model = self.model_wrapper.get_model()

        #self.device = torch.device("cuda:0")
        self.device = torch.device("cpu")

        # load pytorch model
        pt = torch.load(model_name)
        if "module" in pt:
            pt = pt["module"]

        self.model.load_state_dict(pt)
        self.model.to(self.device)
        self.model.eval()

        self.loss = create_loss()

    def do_inference(self, input_tensor, label_tensor = None):
        input_tensor = input_tensor.to(self.device)
        input_tensor = self.model_wrapper.input_preprocess(input_tensor)
        with torch.no_grad():
            output_tensor = self.model(input_tensor)

        output_tensor = self.model_wrapper.output_postprocess(output_tensor)

        loss = 0
        if label_tensor is not None:
            if not isinstance(label_tensor, torch.Tensor):
                label_tensor = torch.tensor(label_tensor)
            label_tensor = label_tensor.to(self.device)
            label_tensor = self.model_wrapper.labels_preprocess(label_tensor)
            loss = self.loss(output_tensor, label_tensor).cpu()
        return output_tensor.cpu(), loss.cpu(), label_tensor.cpu()
