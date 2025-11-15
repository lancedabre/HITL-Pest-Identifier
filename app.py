import streamlit as st
import os
import subprocess
import shutil
from ultralytics import YOLO
from PIL import Image
import io

# --- 1. CONFIGURATION ---
st.set_page_config(layout="wide")

# Paths to your models (Make sure these are in your project folder)
# IMPORTANT: 'yolov8n.pt' should be your *trained v1 model*
# For this demo, let's assume your trained v1 is named 'best_v1.pt'
MODEL_V1_PATH = "best_v1.pt"  # <-- CHANGE THIS to your v1 model file
MODEL_V2_PATH = "best_v2.pt"  # This file will be created after retraining

# Path to the "current best" model the app should use
# Start with v1. You will manually change this to v2 later.
CURRENT_MODEL_PATH = MODEL_V1_PATH 

# Directory to save feedback images for retraining
FEEDBACK_DIR = "pending_retraining_data"
FEEDBACK_IMG_DIR = os.path.join(FEEDBACK_DIR, "images")
FEEDBACK_LABEL_DIR = os.path.join(FEEDBACK_DIR, "labels")

# Your main dataset paths (from your data.yaml)
MAIN_TRAIN_IMG_DIR = "datasets/images/train"
MAIN_TRAIN_LABEL_DIR = "datasets/labels/train"

# Your class names (MUST match your data.yaml)
CLASS_NAMES = ['aphid', 'fruit_flies', 'stink_bugs'] # <-- CHANGE THIS

# Create directories if they don't exist
os.makedirs(FEEDBACK_IMG_DIR, exist_ok=True)
os.makedirs(FEEDBACK_LABEL_DIR, exist_ok=True)

# --- 2. HELPER FUNCTIONS ---

@st.cache_resource  # Caches the model so it doesn't reload
def load_model(model_path):
    """Loads the YOLO model from the specified path."""
    if os.path.exists(model_path):
        return YOLO(model_path)
    else:
        st.error(f"Model file '{model_path}' not found. Please check the CONFIGURATION.")
        return None

def save_feedback(uploaded_file, correct_label_str):
    """Saves the image and a *simplified* label to the feedback directory."""
    
    # 1. Save the image
    img_filename = uploaded_file.name
    img_path = os.path.join(FEEDBACK_IMG_DIR, img_filename)
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # 2. Save the label (This is a simplified example)
    label_filename = os.path.splitext(img_filename)[0] + ".txt"
    label_path = os.path.join(FEEDBACK_LABEL_DIR, label_filename)
    
    if correct_label_str in CLASS_NAMES:
        class_id = CLASS_NAMES.index(correct_label_str)
        
        # Create a dummy label (class_id, x_center=0.5, y_center=0.5, w=1.0, h=1.0)
        # This assumes the pest is the *whole image*
        with open(label_path, "w") as f:
            f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")
        
        return True
    return False

# --- 3. STREAMLIT GUI ---

# Use st.sidebar to create navigation
page = st.sidebar.radio("Navigate", ["Pest Identifier (Farmer)", "Admin Panel (You)"])

# =========== PAGE 1: PEST IDENTIFIER (FOR FARMER) ===========
if page == "Pest Identifier (Farmer)":
    st.title("🌱 Pest Identifier")
    st.write(f"Using Model: **{CURRENT_MODEL_PATH}**")
    st.write("Upload an image of a pest to get an identification.")
    
    st.warning("**Mini-Project Note:** This demo simplifies the 'Human-in-the-Loop' process. When you correct a label, it saves a label for the *entire image*. A real-world app would require a drawing tool to create new, precise bounding boxes.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Convert uploaded file to a PIL Image
        image_bytes = uploaded_file.getvalue()
        pil_image = Image.open(io.BytesIO(image_bytes))

        # Display the uploaded image
        st.image(pil_image, caption="Your Upload", use_column_width=False, width=400)
        
        # Load the model
        model = load_model(CURRENT_MODEL_PATH)
        
        if model:
            ai_guess = "none" # Default
            
            # Run prediction
            with st.spinner("Analyzing pest..."):
                results = model(pil_image)
            
            st.success("Analysis Complete!")
            
            # Process and display results
            if results and len(results) > 0:
                result = results[0] # Get first result object
                
                # Render the image with boxes
                st.write("### AI Detections:")
                result_img_annotated = result.plot() # This returns a BGR numpy array
                st.image(result_img_annotated, caption="AI Detections", use_column_width=False, width=400)

                # Get the top guess for the correction UI
                if len(result.boxes) > 0:
                    top_box = result.boxes[0] 
                    class_id = int(top_box.cls[0])
                    confidence = float(top_box.conf[0])
                    
                    if class_id < len(CLASS_NAMES):
                        ai_guess = CLASS_NAMES[class_id]
                        st.subheader(f"Top Prediction: **{ai_guess.title()}** ({confidence*100:.1f}% sure)")
                    else:
                        st.subheader("AI Prediction: **Unknown Class**")
                        ai_guess = "unknown"
                else:
                    st.subheader("AI Prediction: **No pests found.**")
                    ai_guess = "none"
            else:
                 st.subheader("AI Prediction: **No pests found.**")
                 ai_guess = "none"

            # --- This is the "Human-in-the-Loop" part ---
            st.write("---")
            st.subheader("Is this identification helpful?")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("✅ Yes, looks correct!"):
                    if ai_guess not in ["none", "unknown"]:
                        save_feedback(uploaded_file, ai_guess)
                        st.success("Thank you! Your feedback helps us improve.")
                    else:
                        st.warning("Cannot save 'No Pest' or 'Unknown' feedback.")

            with col2:
                if st.button("❌ No, this is wrong."):
                    st.session_state.show_correction = True

            # If "No" is clicked, show the correction form
            if st.session_state.get('show_correction', False):
                st.write("What is the *correct* pest?")
                correct_label = st.selectbox("Select the correct label:", CLASS_NAMES)
                
                if st.button("Submit Correction"):
                    save_feedback(uploaded_file, correct_label)
                    st.success(f"Got it! We'll retrain the model with this new info. Thank you!")
                    st.session_state.show_correction = False # Hide form


# =========== PAGE 2: ADMIN PANEL (FOR YOU) ===========
elif page == "Admin Panel (You)":
    st.title("🛠️ Admin Panel")
    st.subheader("Model Retraining")
    
    pending_images = os.listdir(FEEDBACK_IMG_DIR)
    st.write(f"You currently have **{len(pending_images)}** new images in your feedback queue.")
    
    if len(pending_images) > 0:
        st.write("Pending files:")
        st.json(pending_images, expanded=False)

    st.info(f"The app is currently using: **{CURRENT_MODEL_PATH}**")
    
    if st.button("RETRAIN MODEL (Create v2)", type="primary"):
        if len(pending_images) == 0:
            st.error("No new images in the feedback queue. Cannot retrain.")
        else:
            st.write("---")
            st.write("Starting retraining process... This may take a while.")
            
            try:
                # --- STEP 1: Move feedback files to main dataset ---
                st.write(f"[1/3] Moving {len(pending_images)} new data files into main training set...")
                moved_count = 0
                for img_file in os.listdir(FEEDBACK_IMG_DIR):
                    shutil.move(os.path.join(FEEDBACK_IMG_DIR, img_file), os.path.join(MAIN_TRAIN_IMG_DIR, img_file))
                    moved_count += 1
                
                for label_file in os.listdir(FEEDBACK_LABEL_DIR):
                    shutil.move(os.path.join(FEEDBACK_LABEL_DIR, label_file), os.path.join(MAIN_TRAIN_LABEL_DIR, label_file))
                
                st.write(f"...Moved {moved_count} images and their labels.")

                # --- STEP 2: Start the YOLO training process ---
                st.write("[2/3] Starting YOLO training command...")
                
                # This command trains a NEW model (v2) using v1's weights as a starting point
                train_command = [
                    "yolo", "train",
                    f"model={MODEL_V1_PATH}",          # Start from v1 weights
                    "data=pest_data.yaml",            # Your main YAML file
                    "epochs=1",                      # You can make this smaller for a quick demo************** CHANGE ***************
                    "imgsz=640",
                    "project=PestProject_Training",   # Save runs to a new folder
                    "name=run_v2"                     # Name this specific run
                ]
                
                # Use subprocess.Popen to run the command in the background
                # We pipe stdout and stderr to a file or variable if we want to display it
                process = subprocess.Popen(train_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                st.write(f"...Training started. This will run in the background.")
                st.info("Check your Colab cell output (where you ran `!streamlit run`) to see the training progress.")
                
                # --- STEP 3: Manually update the model ---
                st.warning("[3/3] **ACTION REQUIRED AFTER TRAINING!**")
                st.write("When training is complete (check Colab output), find the new 'best.pt' file in:")
                st.code("PestProject_Training/run_v2/weights/best.pt")
                st.write(f"1. **Stop** this Streamlit app.")
                st.write(f"2. **Rename** that 'best.pt' file to **'{MODEL_V2_PATH}'** and place it in your main project folder.")
                st.write(f"3. **Edit `app.py`**: Change `CURRENT_MODEL_PATH = \"{MODEL_V1_PATH}\"` to `CURRENT_MODEL_PATH = \"{MODEL_V2_PATH}\"`.")
                st.write("4. **Restart** the app to use your new, smarter model!")
                
            except Exception as e:
                st.error(f"An error occurred: {e}")