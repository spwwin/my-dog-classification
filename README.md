# ITCS476_Digital Image Processing: Dog Classification Project

## Installation

Use the package manager like [pip](https://pip.pypa.io/en/stable/) to install required libraries.

```bash
pip install keras
pip install tensorflow
```
Download a fine-tuned model and extract the file in the same directory where the python files are.

https://drive.google.com/open?id=1IW8_gueI4yKbxfV-CWV_4HGt0tIOnShc

## Usage

*Make sure you run the python file with Python v.3.7+, else some function might not working properly*

To initially fine-tune the model with prepared dataset

```bash
python3 dogclassificationtrain.py
```

To perform a classification based on the existing model and weights

```bash
python3 predict.py --input_dir <dog_image_path>
```

To evaluate the model

```bash
python3 evaluate.py
```

## Contributors

Thanwarat Tanprathumwong #5988176\
Supawit Yangyuenpradorn #5988276
