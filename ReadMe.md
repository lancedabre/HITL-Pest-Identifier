# Human-In-The-Loop Pest Identifier (YOLOv8 & Streamlit)

![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-blueviolet.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-orange.svg)
![Google
Colab](https://img.shields.io/badge/Google-Colab-F9AB00.svg?logo=googlecolab)

This project isn't just an object detector; it's a complete,
self-improving system. It uses **YOLOv8** to identify pests and a
**Streamlit** web app to create a **Human-in-the-Loop (HITL)** feedback
cycle.

When the model makes a low-confidence prediction, a human (the "farmer")
provides the correct label. This feedback is then used to automatically
retrain and improve the model, creating a system that learns from its
own mistakes.

### Demo: The Loop in Action

This is the core value of the project: the model gets smarter with user
feedback.

-   **Model v1 (Initial Guess):** Identifies a pest with
    `74% confidence`.
-   **Human Feedback:** The user confirms or corrects the label.
-   **Retraining:** The user clicks "Retrain," and a `v2` model is
    automatically trained in the background using this new, verified
    data.
-   **Model v2 (After Correction):** The *same image* is now identified
    with `91% confidence`.

------------------------------------------------------------------------

## 🚀 How to Replicate This Project

This entire project is designed to be run from a single **Google Colab
notebook**.

### Project Structure

Your repository should be organized in Google Drive like this:

    /PestDetector/
    ├── app.py                 # The Streamlit web app
    ├── pest_data.yaml         # YOLO dataset configuration file
    ├── production_model.pt    # The starting "v1" model (you must create this)
    ├── datasets/              # Folder for your images and labels
    │   ├── images/
    │   │   ├── train/
    │   │   └── val/
    │   ├── labels/
    │   │   ├── train/
    │   │   └── val/
    └── Pest_Identifier_Colab.ipynb  # Main Colab notebook

------------------------------------------------------------------------

## Part 1: Initial Setup & Training (Model v1)

Before you can run the app, you need a starting "v1" model.

### 1. Set Up Your Colab Environment

Mount your Google Drive and install the YOLOv8 library.

``` bash
from google.colab import drive
drive.mount('/content/drive')

!pip install ultralytics

# Navigate to your project directory
%cd /content/drive/MyDrive/PestDetector/
```

### 2. Prepare Your Dataset

This project requires a dataset in YOLO format.

-   **Get Data:** You can find a "pest" dataset on Roboflow or Kaggle.

-   **Organize:** Place your images and labels in the `datasets/` folder
    as shown in the project structure.

-   **Configure YAML:** Create a `pest_data.yaml` file that points to
    your data. It must contain:

    -   The absolute path to your `datasets` folder in Google Drive.
    -   The relative paths to your `train` and `val` folders.
    -   The number of classes (`nc`) and their names.

    Example `pest_data.yaml`:

    ``` yaml
    path: /content/drive/MyDrive/PestDetector/datasets
    train: images/train
    val: images/val
    nc: 3
    names:
      - 'aphids'
      - 'fruit_flies'
      - 'stink_bugs'
    ```

### 3. Train Your "v1" Model

Run the YOLOv8 training command.

``` bash
!yolo train model=yolov8n.pt data=pest_data.yaml epochs=50 imgsz=640
```

### 4. Create the "Production" Model

After training:

1.  Find `runs/detect/train/weights/best.pt`
2.  Copy it to `/content/drive/MyDrive/PestDetector/`
3.  Rename it to `production_model.pt`

------------------------------------------------------------------------

## Part 2: Running the Human-in-the-Loop App

### 1. Install App Dependencies

``` bash
!pip install streamlit ultralytics pyngrok
```

### 2. Create the `app.py` File

``` python
%%writefile app.py
# Paste your full app.py code here
```

### 3. Add Your `ngrok` Token

``` bash
!ngrok config add-authtoken YOUR_PRIVATE_TOKEN_HERE
```

### 4. Launch the App

``` python
from pyngrok import ngrok
public_url = ngrok.connect(8501)
print(public_url)
!streamlit run app.py
```

------------------------------------------------------------------------

## How to Use the App

1.  Upload an image → model predicts.
2.  Give feedback if prediction is wrong.
3.  Click "Retrain & Activate Model" to generate a smarter model.
4.  Re-upload the same image to see confidence improve.
