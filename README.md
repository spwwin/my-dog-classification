# ITCS476_Digital Image Processing: Dog Classification Project

## Installation

Use the package manager like [pip](https://pip.pypa.io/en/stable/) to install required libraries.

```bash
pip install keras
pip install tensorflow
```

## Usage

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

Thanwarat Tanprathumwong #5988176
Supawit Yangyuenpradorn #5988276