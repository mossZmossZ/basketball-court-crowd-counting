import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
from matplotlib import pyplot as plt, cm as c
from scipy.ndimage.filters import gaussian_filter 
import scipy
import torchvision.transforms.functional as F
from model import CSRNet
import torch
from torchvision import transforms

transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ])

model = CSRNet().cuda()

checkpoint = torch.load('weights.pth', map_location="cuda", weights_only=True)

model.load_state_dict(checkpoint)

img_path = "../IMG/IMG_9872.jpg"

print("Original Image")
plt.imshow(plt.imread(img_path))
plt.show()

img = transform(Image.open(img_path).convert('RGB').resize((720, 480))).cuda()

output = model(img.unsqueeze(0)).cpu()  # Immediately move output to CPU to reduce GPU memory usage
torch.cuda.empty_cache()  # Clear unused GPU memory

print("Predicted Count : ",int(output.detach().cpu().sum().numpy()))
temp = np.asarray(output.detach().cpu().reshape(output.detach().cpu().shape[2],output.detach().cpu().shape[3]))
plt.imshow(temp,cmap = c.jet)
plt.show()