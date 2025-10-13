import torch
from PIL import Image
from torchvision import transforms
import urllib
from models.ctm import ContinuousThoughtMachine as CTM
from tasks.image_classification.imagenet_classes import IMAGENET2012_CLASSES

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CTM.from_pretrained("ciaran-regan-ie/continuous-thought-machines")
    model = model.to(device)
    model.eval()

    url = "https://www.seahorseaquariums.com/image/cache/catalog/Categories%20-%20Freshewater/Coldwater%20Fish/Fantails/Fantail%20Goldfish-2000x2000.jpg"
    filename = "goldfish.jpg"
    target = 1  # Goldfish
    urllib.request.urlretrieve(url, filename)
    image = Image.open(filename).convert("RGB")

    # Preprocess the image
    dataset_mean = [0.485, 0.456, 0.406]
    dataset_std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=dataset_mean, std=dataset_std)
    ])

    input_tensor = transform(image).unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        predictions, certainties, synchronization, pre_activations, post_activations, attention_tracking = model(input_tensor, track=True)

    # Get predictions
    prediction_last = predictions[0, :, -1].argmax(dim=0)
    IMAGENET_CLASS_LIST = list(IMAGENET2012_CLASSES.values())

    print(f"Target Class: {target} = {IMAGENET_CLASS_LIST[target]}")
    print(f"Predicted Class (final): {prediction_last.item()} = {IMAGENET_CLASS_LIST[prediction_last.item()]}")