import torch

class SimpleTest(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            #torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=1))
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            #torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=1))
        self.fc1 = torch.nn.Linear(1024, 1024, bias=True)
    def forward(self, x):
        x = x.view(-1, 1, 1024, 1024)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.fc1(out)
        return out.view(-1, 1024, 1024)