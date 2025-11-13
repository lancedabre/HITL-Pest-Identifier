# Human-In-The-Loop Pest-Identifier (YOLOv8)

This project is a demonstration of a Human-in-the-Loop system for object detection using YOLOv8. The system is designed to identify pests, but more importantly, to use human feedback to correct its own mistakes and improve over time.

## The "Loop"
The goal isn't a perfect model, but a perfect *system* for improvement.

1.  **Predict (Model v1):** A "v1" model is trained on a public dataset. It makes an initial prediction (e.g., "Aphid, 74% confidence").
2.  **Verify (Human):** A human expert looks at this 74% guess.
    * **If correct:** The human "confirms" it, creating a 100% verified label.
    * **If incorrect:** The human "corrects" it (e.g., "That's a Whitefly").
3.  **Re-Train (Model v2):** A new "v2" model is re-trained (fine-tuned) on a dataset that combines the original data *plus* this new batch of human-corrected images.

## Results
By running this loop, we proved the system works:

* **Model v1 (on `test_image.jpg`):** 74% confidence
* **Model v2 (on `test_image.jpg`):** 91% confidence

This demonstrates the power of using human feedback to continually improve the model's blind spots.

## How to Run This Project
1.  Download the Colab Notebook (`.ipynb`).
2.  **Dataset:** Due to file size, the dataset is not included. You can download a public "pest" dataset from Roboflow Universe.
3.  **YAML:** Place your `data.yaml` in the root folder.
4.  Run the notebook cells in order to train v1, simulate the "loop," and train v2.
