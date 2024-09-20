from flask import Flask, render_template, request, url_for, send_from_directory
import os
import uuid
from PIL import Image
import torch
import torchvision.transforms as transforms
from .models import GMM, SegNet, CompNet

# Define Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'media'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Ensure the media directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load models once on startup
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize models
gmm = GMM().to(device)
seg_net = SegNet().to(device)
comp_net = CompNet().to(device)

# Load saved model weights based on GPU available
if device == 'cuda':
    gmm.load_state_dict(torch.load(
        './model_checkpoints/gmm_epoch_150.pth'))
    seg_net.load_state_dict(torch.load(
        './model_checkpoints/seg_net_epoch_150.pth'))
    comp_net.load_state_dict(torch.load(
        './model_checkpoints/comp_net_epoch_150.pth'))
elif device == 'cpu':
    gmm.load_state_dict(torch.load(
        './model_checkpoints/gmm_epoch_150.pth',
        map_location=torch.device('cpu')))
    seg_net.load_state_dict(torch.load(
        './model_checkpoints/seg_net_epoch_150.pth',
        map_location=torch.device('cpu')))
    comp_net.load_state_dict(torch.load(
        './model_checkpoints/comp_net_epoch_150.pth',
        map_location=torch.device('cpu')))

# Set models to evaluation mode
gmm.eval()
seg_net.eval()
comp_net.eval()

# Image transformation for preprocessing input images
transform = transforms.Compose([
    transforms.Resize((256, 192)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


def process_images(body_image_pil, cloth_image_pil):
    """
    Process the body and cloth images using the loaded models to
    produce a composite image.
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


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in\
        app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    """Render the upload form."""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    # Check if files are in the request
    if 'body_image' not in request.files or 'cloth_image' not in request.files:
        return {"error": "Files are missing"}, 400

    body_image = request.files['body_image']
    cloth_image = request.files['cloth_image']

    # Check if files are allowed
    if body_image and allowed_file(body_image.filename) and cloth_image and allowed_file(cloth_image.filename):
        # Generate unique filenames
        body_image_name = f"{uuid.uuid4()}." + \
            body_image.filename.rsplit('.', 1)[1].lower()
        cloth_image_name = f"{uuid.uuid4()}." + \
            cloth_image.filename.rsplit('.', 1)[1].lower()

        # Save the images
        body_image_path = os.path.join(app.config['UPLOAD_FOLDER'],
                                       body_image_name)
        cloth_image_path = os.path.join(app.config['UPLOAD_FOLDER'],
                                        cloth_image_name)
        body_image.save(body_image_path)
        cloth_image.save(cloth_image_path)

        # Open images with PIL
        body_image_pil = Image.open(body_image_path)
        cloth_image_pil = Image.open(cloth_image_path)

        # Process the images
        processed_image = process_images(body_image_pil, cloth_image_pil)

        # Save processed image
        processed_image_name = f"{uuid.uuid4()}.png"
        processed_image_path = os.path.join(app.config['UPLOAD_FOLDER'],
                                            processed_image_name)
        processed_image.save(processed_image_path)

        # Redirect to result page
        return {"redirect": url_for('result',
                                    filename=processed_image_name)}, 200

    return {"error": "Invalid file type"}, 400


@app.route('/result/<filename>')
def result(filename):
    return render_template('result.html', filename=filename)


@app.route('/media/<filename>')
def media(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(port=5000)
