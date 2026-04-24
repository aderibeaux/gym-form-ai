## Requirements

Python 3.9+

Required libraries are listed in requirements.txt.

## Installation

Clone the repository:

git clone <repo-url> ******************
cd gym-form-ai

Install dependencies:
pip install -r requirements.txt


## Running the Project
Run the real-time bulgarian split squat analyzer:
python src/realtime_predict.py

The webcam will open and begin detecting pose landmarks.

Perform Bulgarian split squats in front of the camera.

Press **ESC** to stop recording.

The system will output predicted squat form and feedback for each repetition.

## Data Generation

To record pose data:
python src/pose_test.py

To generate rep features from pose data:
python src/plot_motion.py

To build the labeled dataset:
python src/build_dataset.py

## Model Training

Train the baseline model:
python src/train_model.py

Compare models:
python src/compareModels.py

Run cross-validation:
python src/cross_val_models.py

Tune Random Forest hyperparameters:
python src/hyperparam_tune_forest.py

Evaluate the final model:
python src/analyze_random_forest.py

