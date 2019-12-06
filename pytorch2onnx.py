import torch.onnx
import torchvision
import resnext 
from torch.autograd import Variable

# Standard ImageNet input - 3 channels, 224x224,
# values don't matter as we care about network structure.
# But they can also be real inputs.

def get_mean():
    return [114.7748, 107.7354, 99.4750]

mean = get_mean()
arch = 'resnext-101'
sample_size = 112
sample_duration = 16
n_classes = 400

model = resnext.resnet101(num_classes=n_classes, sample_size=sample_size, sample_duration=sample_duration, shortcut_type='B')
dummy_input = torch.randn(64, 3, 7, 7, 7)

model_data = torch.load('C:\\Users\\ribeirfi\\git\\videoextractor\\weights\\resnext-101-kinetics.pth',map_location=torch.device('cpu'))

print (model_data['arch'])
assert arch == model_data['arch']
model.load_state_dict(model_data['state_dict'], strict=False)
model.eval()


# Invoke export
torch.onnx.export(model, dummy_input, "resnet-101-kinetics.onnx")