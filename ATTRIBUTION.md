# AI Assistance

In general ChatGPT was used for assistance with:
- debugging python code/ making improvements on efficiency
- explaining machine learning concepts
- providing a checklist of what to complete and general explanations of how
- aligning my project with rubric items

Specific uses:
- providing a general structure for the README, SETUP, and ATTRIBUTION
- In analyze_random_forest the code on importance of features and the visualization
- fixing small errors in build_dataset
- the creation of the models dictionary in compareModels and a general structure of what the file should be
- The use of Pipeline in cross_val_model
- structure for hyperparam_tune_forest
- model_comparison_chart
- calculate_angle and angle_to_vertical functions in plot_motion as well as the smoothing in the plotting
- Most of pose_test came from the documentation but used some AI for adjusting it to what I wanted
- test_env was completely AI to test if environment was set up correctly

# External Libraries

This project uses the following open-source libraries:

- OpenCV
- MediaPipe
- NumPy
- pandas
- scikit-learn
- SciPy
- matplotlib
- joblib

# MediaPipe Pose

Pose detection uses Google's MediaPipe Pose framework.

https://github.com/google-ai-edge/mediapipe

# Machine Learning Algorithms

Machine learning models were implemented using scikit-learn.

https://scikit-learn.org/

# Dataset

The dataset used in this project was collected by recording Bulgarian split squat videos and extracting pose landmarks using MediaPipe (with multiple different test subjects).

No external datasets were used.

