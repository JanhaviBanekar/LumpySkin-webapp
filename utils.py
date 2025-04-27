from PIL import Image
import torchvision.transforms as transforms
import torch

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

def predict_image(image_path, model):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        predicted = torch.argmax(output, dim=1).item()

    if predicted == 1:
        return "✅ NORMAL", "The skin appears healthy. No treatment required."
    else:
        return "⚠️ INFECTED", "Lumpy skin detected. Consult a veterinarian and isolate the animal."
