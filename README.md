# RGBT Crack Detection

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![Static Badge](https://img.shields.io/badge/made_with-Pytorch-orange)](https://pytorch.org/)
[![Static Badge](https://img.shields.io/badge/made_with-streamlit-red)](https://streamlit.io/)

A simple CNN model for multi-spectral image based crack detection.

## Setup

You will need to install all the dependencies by launching the command:  (it is recommended to use a virtual environment)

    pip install -r requirements.txt


If you plan to use wandb as a logging tool, you will need to install it:

    pip install wandb
    

## Usage

### Dataset

We provide the tool used to create the dataset. We will need the RGBT orthophoto and a similar-sized image with the cracks annotated in black with a white background. After modifying the path of the files at the end of the Python script, you will need to launch the command:

    python dataset.py

### Training

To train the model, you will need to launch the command:

    python train.py 

You can modify multiple parameters by changing the constant values in the train.py file.

### Inference

We provide a simple user interface to test the model on your own images using the [streamlit](https://streamlit.io/) library. To launch it, you will need to launch the command:

    streamlit run Classifier.py

Please note that the given models were trained on a specific and small dataset. As such, they do not display good generalizability. We recommend that you train your own model using the train.py script.
