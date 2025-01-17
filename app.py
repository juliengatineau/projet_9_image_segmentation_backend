from flask import Flask, request, send_file
import logging
import os
from PIL import Image
import numpy as np
from matplotlib import colors
import requests
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

import torch
from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights

# Afficher le chemin de travail actuel
logging.info(f"Chemin de travail actuel : {os.getcwd()}")

# --------------------------------------------------------------------
# DOWNLOAD MODEL
# --------------------------------------------------------------------

model_dir = './model'
try:
    os.makedirs(model_dir, exist_ok=True)
    logging.info(f"Dossier '{model_dir}' créé avec succès ou déjà existant.")
    # Lister les fichiers dans le répertoire actuel
    logging.info(f"Contenu du répertoire actuel : {os.listdir(os.getcwd())}")
except Exception as e:
    logging.error(f"Erreur lors de la création du dossier '{model_dir}': {e}")


# Fonction pour télécharger le modèle depuis Github
def download_model_from_github(repo_owner, repo_name, release_tag, asset_name, destination):
    url = f'https://api.github.com/repos/{repo_owner}/{repo_name}/releases/tags/{release_tag}'
    response = requests.get(url)
    response.raise_for_status()
    release = response.json()
    asset = next(item for item in release['assets'] if item['name'] == asset_name)
    download_url = asset['browser_download_url']
    
    if not os.path.exists(destination):
        response = requests.get(download_url, stream=True)
        response.raise_for_status()
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logging.info(f"Modèle téléchargé avec succès à '{destination}'")
    else:
        logging.info(f"Le modèle existe déjà à '{destination}'")

# Informations sur la release GitHub
repo_owner = 'juliengatineau'
repo_name = 'projet_9_backend'
release_tag = 'v1.0.0'
asset_name = 'sernet_model.pth'
model_path = os.path.join(model_dir, asset_name)

# Télécharger le modèle depuis la release GitHub
download_model_from_github(repo_owner, repo_name, release_tag, asset_name, model_path)

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

    # Convert the PIL image to a BytesIO object
    img_io = BytesIO()
    mask.save(img_io, 'PNG')
    img_io.seek(0)

    # Return the image as a response
    return send_file(img_io, mimetype='image/png', as_attachment=True, download_name=predicted_mask_filename)
        

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)