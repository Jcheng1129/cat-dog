import gradio as gr
import torch
import cv2
import torchvision.transforms as T
from torchvision.models import alexnet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = alexnet(weights=None)
model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, 2)
model.load_state_dict(torch.load("./best_model.pth", map_location=device))
model = model.to(device)
model.eval()

def gradio_predict(input_image):
    img = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    trans = T.Compose([
        T.ToPILImage(),
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    tensor = trans(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)
        prob = torch.softmax(out, dim=1)
        cat_p = prob[0][0].item()
        dog_p = prob[0][1].item()
    return {
        "cat": round(cat_p,4),
        "dog": round(dog_p,4)
    }

demo = gr.Interface(
    fn=gradio_predict,
    inputs=gr.Image(type="numpy", label="上传图片 / 拍照", sources=["upload","webcam"]),
    outputs=gr.Label(num_top_classes=2, label="分类结果"),
    title="猫狗图像分类",
    description="上传图片或调用摄像头拍照，识别猫或者狗"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
