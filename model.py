import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from torchvision.transforms import functional as F

#Class names
class_names = [
    "Acinetobacter baumannii (aba)",
    "Bacillus cereus (bce)",
    "Burkholderia cepacia (bcp)",
    "Corynebacterium striatum (cst)",
    "Enterococcus casseliflavus (eca)",
    "Escherichia coli (eco)",
    "Enterococcus faecium (efa)",
    "Enterobacter cloacae (enc)",
    "Klebsiella pneumoniae (kpn)",
    "Morganella morganii (mmo)",
    "Pseudomonas putida (ppu)",
    "Streptococcus agalactiae (sag)",
    "Staphylococcus aureus (sau)",
    "Staphylococcus epidermidis (sep)",
    "Salmonella enterica subsp. enterica (ses)",
    "Serratia marcescens (sma)",
    "Streptococcus pneumoniae (spn)",
    "Streptococcus pyogenes (spy)",
    "Stenotrophomonas maltophilia (stm)"
]

#Resize + padding

class ResizeWithPadding:
  # Store the desired final image size
  def __init__(self,target_size):
    self.target_size = target_size

   # Apply the resize + padding operation to an image
  def __call__(self,image):
      # Get the original image dimensions
    original_width,original_height = image.size
      # Get the required final dimensions
    target_width,target_height = self.target_size

      # Calculate the scale that keeps the entire image
      #  The minimum scale is used to ensure that both the width and height fit within the 224 × 224 target size without cropping.

    scale = min(target_width/original_width, target_height/original_height)

      # Calculate the new dimensions after proportional resizing
    resized_width = int(original_width * scale)
    resized_height =int(original_height * scale)

     # Resize the image while preserving its aspect ratio
    image =F.resize(image,(resized_height,resized_width))
    # Calculate the total padding needed on each dimension
    total_horizontal_padding = target_width - resized_width
    total_vertical_padding = target_height - resized_height
    # Divide the horizontal padding between left and right
    padding_left = total_horizontal_padding//2
    padding_right = total_horizontal_padding - padding_left

    # Divide the vertical padding between top and bottom
    padding_top = total_vertical_padding // 2
    padding_bottom = total_vertical_padding - padding_top

     # Add the calculated padding
     # fill=0 means the padding area will be black

     #For torchvision.transforms.functional.pad(), the padding list is written in this order:[left, top, right, bottom]
    image = F.pad(image, [padding_left,padding_top,padding_right,padding_bottom], fill =0)
      # Return the final 224 × 224 image
    return image



# Create the resize + padding transform with a target size of 224 × 224
resize_and_pad = ResizeWithPadding((224, 224))

#Image transformation
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]



image_transform = transforms.Compose([
    resize_and_pad,
    transforms.ToTensor(),
    transforms.Normalize(mean=mean,std=std)

])

#Model 1
model1 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model1.fc = nn.Linear(model1.fc.in_features, 19)

model1.load_state_dict(
    torch.load(
      "weights/Model1_ExperimentA_weights.pth",
        
        map_location="cpu"
    )
)

model1.eval()

#Model 2
model2 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model2.fc = nn.Linear(model2.fc.in_features, 19)

model2.load_state_dict(
    torch.load(
        "weights/Model2_ExperimentB_weights.pth",
        map_location="cpu"
    )
)

model2.eval()

#Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model1 = model1.to(device)
model2 = model2.to(device)

print(device)

#Prediction function
def predict_image(image, model):

    image = image.convert("RGB")

    image = image_transform(image)

    image = image.unsqueeze(0)

    image = image.to(device)

    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)

    predicted_name = class_names[predicted_class.item()]
    confidence = confidence.item()

    return predicted_name, confidence