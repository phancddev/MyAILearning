import torch
from PIL import Image

def test_model(model, image_path, model_path, transform):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load mô hình
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model = model.to(device)
    model.eval()
    
    # Load ảnh
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)
    
    # Dự đoán
    output = model(image)
    prediction = "Apple" if output.item() > 0.5 else "Not Apple"
    print(f"Prediction: {prediction}")