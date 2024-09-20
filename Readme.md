# Virtual Try-On Model (VITON) - Project Overview

This project aims to develop a Virtual Try-On system using deep learning models. The system allows users to see how a piece of clothing would look on a person given a body image and a clothing image. The implementation uses three primary models: GMM (Geometric Matching Module), SegNet (Segmentation Network), and CompNet (Composition Network) to achieve high-quality results.

## Project Goals

- **Objective**: The goal is to build a virtual try-on system that aligns a given clothing image onto a body image, producing a realistic composite image that shows how the clothing would look when worn.
- **Models Used**:
  1. **GMM (Geometric Matching Module)**: Warps the clothing image to align it with the body image using a Spatial Transformer Network (STN) with Thin-Plate Spline (TPS) transformation.
  2. **SegNet (Segmentation Network)**: Generates a segmentation mask of the person to identify different regions like the upper body, arms, and legs, ensuring the clothing is correctly positioned.
  3. **CompNet (Composition Network)**: Combines the warped clothing and the body image based on the segmentation mask to create the final composite image.

## Flask Web Application

The project includes a **Flask web application** that provides a user-friendly interface for testing the trained models with external body and cloth images. The web app consists of two main pages:

1. **Upload Page**: The first page contains a form with two input fields where users can upload a body image and a cloth image. Upon selecting the images correctly, the form data is submitted using AJAX, and a loading icon appears on the screen indicating the process has started.
  
2. **Processing and Redirect**: The images are processed by the models to generate the composite image. During this time, the user sees a loading indicator.

3. **Result Page**: Once the composite image is created and saved, the user is redirected to the second page, where the composite image is displayed in the center of the screen.

### Key Features of the Flask Web App:

- **AJAX-Based Submission**: Image files are uploaded and processed asynchronously using AJAX, enhancing user experience by preventing page reloads.
- **Loading Indicator**: A loading animation is shown while the server processes the images.
- **Efficient Resource Use**: Models are loaded only once when the Flask app starts, optimizing resource use and speeding up subsequent requests.

## Dataset Structure

We used the **VITON_resize** dataset, which is a resized and lighter version of the original VITON dataset to efficiently manage computational resources. The directory structure of the dataset is as follows:

```
viton_resize/
│
├── train/
│   ├── cloth/            # Contains cloth images used for training
│   ├── cloth-mask/       # Binary masks for the cloth images
│   ├── image/            # Body images of people wearing plain clothing
│   ├── image-parse/      # Segmentation masks of the body images
│   └── pose/             # Pose keypoints for the body images
│
└── test/
    ├── cloth/            # Contains cloth images used for testing
    ├── cloth-mask/       # Binary masks for the cloth images
    ├── image/            # Body images of people wearing plain clothing
    ├── image-parse/      # Segmentation masks of the body images
    └── pose/             # Pose keypoints for the body images
```

- **Cloth**: Contains the images of the clothing that we want to fit on the person.
- **Cloth-mask**: Contains binary masks corresponding to the clothing images to identify the clothing area.
- **Image**: Contains images of people that will wear the clothing.
- **Image-parse**: Contains segmentation masks of the person images, separating different body parts like torso, arms, etc.
- **Pose**: Contains JSON files with pose keypoints to provide detailed body positioning information.

## Why Use VITON_resize?

We chose to use the **VITON_resize** dataset because it consumes fewer resources compared to the full-size dataset, making it feasible to train on standard computational setups. However, this does not compromise the quality significantly.

Additionally, to manage resources more efficiently, we experimented with using only 30% of the training and testing datasets. This helps in reducing training time and computational load while still maintaining a representative dataset for model training.

> **Note**: The dataset is not publicly available. Therefore, it is not included in this repository. The dataset and implementation details are based on the research paper: [VITON: An Image-Based Virtual Try-On Network](https://openaccess.thecvf.com/content_cvpr_2018/papers/Han_VITON_An_Image-Based_CVPR_2018_paper.pdf).

## Model Training

If you need to train the models, you should first locate the **VITON_resize** dataset from the internet, set it up correctly, and then train the models using the scripts provided in the `train_model` directory.


### Training Details

The models were trained for 150 epochs using the VITON dataset to achieve optimal results. The training process utilized all available data, ensuring high-quality outputs for the Virtual Try-On system.


### Advantage of Our Training Approach

Our training approach is designed with flexibility and efficiency in mind. The code allows models to be saved at any point during training, and these saved models can be reloaded later to continue training from the exact state where they were left off. This feature is particularly beneficial for:

- **Longer Training Sessions**: Training deep learning models can be time-consuming and resource-intensive. The ability to save and reload models means you don't have to start from scratch every time, allowing you to manage longer training sessions effectively.
  
- **Incremental Improvements**: By continuing training from a saved state, you can experiment with additional epochs or adjust learning rates and other hyperparameters to further improve the model’s performance without losing the progress made in earlier sessions.

- **Resource Management**: If training needs to be paused due to resource constraints (e.g., limited GPU availability), you can easily resume training later without any loss of work.

- **Fine-Tuning**: This approach also facilitates fine-tuning the models on new data or specific cases by loading the previously trained model and continuing training for additional epochs.

By saving and reloading models, our approach enables a seamless workflow for iterative model enhancement, optimizing both time and computational resources.


## Running the Project

### Setting Up the Environment

1. **Create a Virtual Environment**: Before running the project, create a virtual environment and install the required libraries using `requirements.txt`:

   ```bash
   python -m venv vton_env
   source vton_env/bin/activate  # On Windows: vton_env\Scripts\activate
   pip install -r requirements.txt
   ```

### Running the Flask Web App

1. **Download Pre-Trained Models**: To run the web application, you need the pre-trained models. Download the models from the Google Drive link below, unzip them, and place them in the `web_app` directory:

   - **Google Drive Link for Pre-trained Models**: [Download Models](#)

2. **Run the Flask App**: Navigate to the `web_app` directory and start the Flask application:

   ```bash
   cd web_app
   python app.py
   ```

3. **Using the Web Interface**:
   - Open the browser and go to `http://localhost:5000`.
   - Upload the body and cloth images using the form on the first page.
   - Wait until the composite image is generated (indicated by the loading icon), and you will be redirected to the result page where the final composite image is displayed.

## Acknowledgments

This project is inspired by and based on the research work presented in the paper: [VITON: An Image-Based Virtual Try-On Network](https://openaccess.thecvf.com/content_cvpr_2018/papers/Han_VITON_An_Image-Based_CVPR_2018_paper.pdf).

## License
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.