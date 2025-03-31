from flask import Flask,render_template,request
import torch
import torchvision
from torchvision.transforms import transforms
from PIL import Image
import os
import torchvision.models as models
from torch import nn

app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(
        in_features=model.fc.in_features,
        out_features=3
    )
    model.load_state_dict(torch.load('model/model_1.pth',map_location=torch.device(device)))
    model.to(device)
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

def predict(img_path):
    img = Image.open(img_path)
    img = transform(img)
    img = img.unsqueeze(dim=0).to(device)
    with torch.inference_mode():
        y_pred = model(img)
        y_pred_prob = torch.softmax(y_pred,dim=1)
        y_label = torch.argmax(y_pred_prob,dim=1)

    class_labels = ["Healthy", "Powdery", "Rust"]
    return class_labels[y_label.item()]


@app.route("/",methods=['GET','POST'])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded"
        
        file = request.files["file"]
        if file.filename == "":
            return "No file selected"
        
        file_path = os.path.join("static/uploads",file.filename)
        file.save(file_path)

        prediction = predict(file_path)

        return render_template("result.html",prediction=prediction,file_path="static/uploads/"+file.filename    )
    
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)