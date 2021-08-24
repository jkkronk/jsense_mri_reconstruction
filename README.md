# JSENSE_numpy
This repository is a simple implementation of JSENSE MRI reconstruction using numpy. JSENSE is a joint reconstruction algorithm of images and coil sensitivity maps are performed. The sensitivity maps are parameterized using polynomial function and solved using least squares. 

[Link to original paper](https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.21245)

The test image is undersampled and reconstruction in 'run_jsense.py' and initialisation of polynomial basis function together with least squares solution to reconstruct the sensitivity maps are located in 'sense_estimation.py'.

Pytorch implementation in 'sense_estimation.py'. Not tested yet.

## Dependencies and Installation
Written for Python 3.8.5. [Sigpy](https://sigpy.readthedocs.io/en/latest/index.html) is used for Espirit estimation of initial sensitivity maps, and for comparable baseline. 

```bash
pip install -r requirements.txt
```

`pip` will install all dependencies.

## Usage
To run with the test MRI slice run the main file as:
```bash
python run_jsense.py
```

## Data
Example data is from FastMRI dataset found [here](https://fastmri.org/). Brain MRI aquired in flair axial view.

## Other
WIP, not fully tested.
Feel free to reach out to jonatank@ee.ethz.ch for any questions.
