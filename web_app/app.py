from flask import Flask, render_template, request, redirect, url_for, send_file
from PIL import Image
import torch
import torchvision.transforms as transforms
import io
import os

# Define Flask app
app = Flask(__name__)

# Load models once on startup
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize models
gmm = GMM().to(device)
seg_net = SegNet().to(device)
comp_net = CompNet().to(device)

# Load saved model weights
gmm.load_state_dict(torch.load('./model_checkpoints/gmm_epoch_10.pth'))
seg_net.load_state_dict(torch.load('./model_checkpoints/seg_net_epoch_10.pth'))
comp_net.load_state_dict(torch.load('./model_checkpoints/comp_net_epoch_10.pth'))

# Set models to evaluation mode
gmm.eval()
seg_net.eval()
comp_net.eval()

# Image transformation for preprocessing input images
transform = transforms.Compose([
    transforms.Resize((256, 192)),
    transforms.ToTensor(),
])


def process_images(body_image_pil, cloth_image_pil):
    """
    Process the body and cloth images using the loaded models to produce a composite image.
    """
    # Preprocess input images
    body_image = transform(body_image_pil).unsqueeze(0).to(device)
    cloth_image = transform(cloth_image_pil).unsqueeze(0).to(device)

    # Step 1: Warping the cloth using GMM
    with torch.no_grad():
        warped_cloth = gmm(body_image, cloth_image)

    # Step 2: Generating the segmentation mask using SegNet
    seg_input = torch.cat([body_image, warped_cloth], dim=1)
    with torch.no_grad():
        pred_mask = seg_net(seg_input)

    # Step 3: Composing the final image using CompNet
    with torch.no_grad():
        composite_img = comp_net(body_image, warped_cloth, pred_mask)

    # Convert the output tensor to PIL image
    composite_image = composite_img.squeeze(0).cpu().detach()
    composite_image_pil = transforms.ToPILImage()(composite_image)

    return composite_image_pil


@app.route('/')
def index():
    """Render the upload form."""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    """Handle the upload and redirect to the waiting page."""
    if 'body_image' not in request.files or 'cloth_image' not in request.files:
        return "Please upload both body and cloth images."

    # Save uploaded images temporarily
    body_image = request.files['body_image']
    cloth_image = request.files['cloth_image']

    # Save images to static folder
    body_image.save(os.path.join('static', 'body_image.jpg'))
    cloth_image.save(os.path.join('static', 'cloth_image.jpg'))

    # Redirect to waiting page
    return redirect(url_for('waiting'))


@app.route('/waiting')
def waiting():
    """Render the waiting page."""
    # Process images and redirect to result page when done
    body_image_pil = Image.open(os.path.join('static', 'body_image.jpg')).convert('RGB')
    cloth_image_pil = Image.open(os.path.join('static', 'cloth_image.jpg')).convert('RGB')
    composite_image_pil = process_images(body_image_pil, cloth_image_pil)

    # Save the composite image
    composite_image_pil.save(os.path.join('static', 'composite_image.jpg'))

    # Redirect to the result page
    return redirect(url_for('result'))


@app.route('/result')
def result():
    """Render the result page with the composite image."""
    return render_template('result.html', image_url='/static/composite_image.jpg')


if __name__ == '__main__':
    app.run(debug=True, port=8000)
