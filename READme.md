# DES423-Heart-Disease-Prediction

Making Prediction for knowledge and for term project in Applied Machine Learning course (DES423) Only for educational purpose.

**NOTE We use the data set called `heart_all_clean_binary.csv` because we want to predict the presence or absence (which means like have or not have) of the heart disease in the patient, and detect possible disease, about severity `num` column in the dataset (หรือ stage of disease or in thai ระดับความรุนแรงของ patient นั้น). So we can use other one which is `heart_all_clean.csv` but for first screenning (which means like first-pass check of the stage). Using binary one can predict who has it or not is for this purpose and we could modify and configure to predict the severity with other dataset.

# Installation Guide
This project uses Python 3.11 (PyCaret 3.x doesn’t support 3.13 reliably at the moment)
- I advise you to use CONDA or Virtual Environment (venv) to use python 3.11 version (Steps down below)

These command will guide you how to install, change python version, and how to run virutal environment.

- winget install --id Python.Python.3.11 --source winget // This command install python version 3.11 in to your machine.
- py -0p // List python versions path
- py -3.11 -V // Select python 3.11 version path
- py -3.11 -m venv .venv311 // create virtual environment 3.11
- .venv311\Scripts\activate // Activate virtual environment
- pip install -r requirements.txt // Install dependencies and packages
- python {filename}.py // To run the python file

# Steps to run a file


After all installation done, I advise to run `Trainmodel.py` first, because it would generate a csv with prediction and metrics which will be used in `confusionMatrix.py` and `featureImportance.py`

# Outputs
The results are in the ouputs folder, which if it doesn't exist then run the files in steps like the above instruction.
The folders are seperated for each python file (Output of the files that are generated)
