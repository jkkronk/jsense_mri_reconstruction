# JSENSE_numpy
This repository is an simple implementation of JSENSE MRI reconstruction using numpy. JSENSE is a joint reconstruction algorithm of images and coil sensitivity maps are performed. The sensitivity maps are parameterize using polynomial function and solved using least squares. 

[Link to original paper](https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.21245)

## Dependencies and Installation
Written for Python 3.8.5

```bash
pip install -r requirements.txt
```

`pip` will install all package dependencies.

## Usage
To run with the test MRI slice run the main file as:
```bash
python run_jsense.py
```

## Data
Example data is from FastMRI dataset found [here](https://fastmri.org/).

## Other
Feel free to reach out to jonatank@ee.ethz.ch for questions.
