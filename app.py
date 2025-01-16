from flask import Flask, request, send_file
import logging
import os
from PIL import Image
import numpy as np
from matplotlib import colors
import requests

# Configure logging
logging.basicConfig(level=logging.DEBUG)

import torch
from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights


# --------------------------------------------------------------------
# DOWNLOAD MODEL
# --------------------------------------------------------------------

def download_model_from_drive(drive_url, destination):
    if not os.path.exists(destination):
        response = requests.get(drive_url, stream=True)
        response.raise_for_status()
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

os.makedirs('./model', exist_ok=True)

# URL du modèle sur Google Drive
drive_url = 'https://drive.google.com/file/d/10I6Biwyvv8Xc2Vtx6YScaZZDTfoznbkn/view?usp=sharing'
model_path = "./model/sernet_model.pth"

# Télécharger le modèle depuis Google Drive
download_model_from_drive(drive_url, model_path)


# --------------------------------------------------------------------
# VARIABLES
# --------------------------------------------------------------------

# Path to the Keras model
model_path = "./model/sernet_model.pth"

# Load the Keras model
preprocess = DeepLabV3_ResNet101_Weights.DEFAULT.transforms()
sernet_model = torch.load(model_path, map_location=torch.device('cpu'))
sernet_model.eval()


# --------------------------------------------------------------------
# FUNCTIONS
# --------------------------------------------------------------------

def generate_img_from_mask_sernet(ref):
    # Palette de couleurs pour chaque catégorie
    colors_palette=['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

    id2category = {0: 'void',
                   1: 'flat',
                   2: 'construction',
                   3: 'object',
                   4: 'nature',
                   5: 'sky',
                   6: 'human',
                   7: 'vehicle'}
    
    # Initialiser l'image de sortie
    img_seg = np.zeros((ref.shape[0], ref.shape[1], 3), dtype='float')

    # Assigner les canaux RGB
    for cat in id2category.keys():
        mask = (ref == cat)
        img_seg[:, :, 0] += mask * colors.to_rgb(colors_palette[cat])[0]
        img_seg[:, :, 1] += mask * colors.to_rgb(colors_palette[cat])[1]
        img_seg[:, :, 2] += mask * colors.to_rgb(colors_palette[cat])[2]

    # Convertir l'image en type uint8 pour l'affichage
    img_seg = (img_seg * 255).astype(np.uint8)

    return img_seg



# --------------------------------------------------------------------
# APP
# --------------------------------------------------------------------

app = Flask(__name__)
@app.route('/predict', methods=['POST'])
def predict():

    logging.info('----------------------------predict-backend---------------------------')
    image_path = request.files['image']  # Récupération de l'image
    predicted_mask_filename = request.form.get('predicted_mask_filename')

    # Preprocess
    image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    # Predict
    with torch.no_grad():
        output = sernet_model(input_batch)
    output = output["out"]

    # Postprocess
    predicted_mask = output.argmax(dim=1).cpu().numpy()[0]
    mask = Image.fromarray(generate_img_from_mask_sernet(predicted_mask))

    # Return the image as a response
    return send_file(mask, mimetype='image/png', as_attachment=True, download_name=predicted_mask_filename)
        

if __name__ == '__main__':
    app.run(debug=True, port=8001)