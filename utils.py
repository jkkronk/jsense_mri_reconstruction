# CODE INSPIRED BY FASTMRI: https://github.com/facebookresearch/fastMRI

import numpy as np

def choose_acceleration(seed, R):
    rng = np.random.RandomState()
    rng.seed(seed)
    if R == 2:
        center_fraction = 0.16
    elif R == 3:
        center_fraction = 0.12
    elif R == 4:
        center_fraction = 0.08
    elif R == 8:
        center_fraction = 0.04
    else:
        print('EXIT: Undersampling ratio not implemented____')
        exit()

    acceleration = R
    return center_fraction, acceleration

def generate_US_pattern(shape, R=4 ,seed=1):
        """
            Args:
                shape: The shape of the mask to be created. The shape should have
                    at least 3 dimensions. Samples are drawn along the second last
                    dimension.
                seed: Seed for the random number generator. Setting the seed
                    ensures the same mask is generated each time for the same
                    shape. The random state is reset afterwards.

            Returns:
                A mask of the specified shape.
            """
        if len(shape) < 3:
            raise ValueError("Shape should have 3 or more dimensions")

        rng = np.random.RandomState()
        rng.seed(seed)
        center_fraction, acceleration = choose_acceleration(seed, R)
        num_cols = shape[2]
        num_low_freqs = int(round(num_cols * center_fraction))

        # create the mask
        mask = np.zeros(num_cols, dtype=np.float32)
        pad = (num_cols - num_low_freqs + 1) // 2
        mask[pad: pad + num_low_freqs] = True

        # determine acceleration rate by adjusting for the number of low frequencies
        adjusted_accel = (acceleration * (num_low_freqs - num_cols)) / (
                num_low_freqs * acceleration - num_cols
        )
        offset = rng.randint(0, round(adjusted_accel))

        accel_samples = np.arange(offset, num_cols - 1, adjusted_accel)
        accel_samples = np.around(accel_samples).astype(np.uint)
        mask[accel_samples] = True

        # reshape the mask
        mask = np.repeat(mask[np.newaxis, :], shape[1], axis=0)
        mask = np.repeat(mask[np.newaxis, :, :], shape[0], axis=0)

        return mask, num_low_freqs

def rmse(pred, gt):
    return np.linalg.norm(pred.flatten() - gt.flatten()) ** 2 / np.linalg.norm(gt.flatten()) ** 2
