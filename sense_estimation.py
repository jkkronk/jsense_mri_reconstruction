import numpy as np

def normalize_basis(basis_funct):
    # Code from Melanie Gaillochet https://github.com/Minimel/MasterThesis_MRI_Reconstruction
    """
    We return a list of orthonormal bases using Gram Schmidt algorithm
    :param X: list of basis we would liek to orthonormalize
    :return:
    """
    matrix = np.array(basis_funct).T

    orthonormal_basis, _ = np.linalg.qr(matrix)

    orthonormal_basis = orthonormal_basis.T
    return orthonormal_basis
    
def create_basis_functions(num_rows, num_cols, max_order, orthonormalize=True, show_plot=False):
    # Code from Melanie Gaillochet https://github.com/Minimel/MasterThesis_MRI_Reconstruction
    """
    We are creating 2D polynomial basis functions
    (ie: a0 + a1*x + a2*y + a3*xy + a4*x^2 + a5*x^2*y + a6*y^2x +....)
    :param num_rows:
    :param num_cols:
    :param image: x estimate of the reconstruction image (if we want the base
    to be multiplied: E^H E bx)
    :return: a list with all basis [(m x n)] arrays
    """
    # We take all the values between 0 and 1, in the x and y axis
    y_axis = np.linspace(-1, 1, num_rows)
    x_axis = np.linspace(-1, 1, num_cols)

    X, Y = np.meshgrid(x_axis, y_axis, copy=False)
    X = X.flatten().T
    Y = Y.flatten().T

    basis_funct = np.zeros(((max_order+1)**2, num_cols*num_rows))
    i = 0
    for power_x in range(0, max_order + 1):
        for power_y in range(0, max_order + 1):
            current_basis = X ** power_x * Y ** power_y
            basis_funct[i,:] = current_basis
            i += 1

    # We normalize the basis function
    if orthonormalize:
        basis_funct = normalize_basis(basis_funct)

    return basis_funct.reshape((max_order+1)**2, num_rows, num_cols)


def sense_estimation_ls(Y, X, basis_funct, uspat):
    """
    Estimation the sensitivity maps for MRI Reconstruction using polynomial basis functions 
    :param data: y (undersampled kspace) [c x n x m]
    :param X: predicted reconstruction estimate [c x n x m]
    :param max_basis_order:
    :return coefficients: Least squares coefficients for basis functions
    """
    num_coils, sizex, sizey = Y.shape
    num_coeffs = basis_funct.shape[0]
    coeff_coils = np.zeros((num_coils, num_coeffs), dtype=complex)

    for i in range(num_coils):
        Y_i = Y[i].reshape(sizex*sizey)
        A_i = (uspat[i, np.newaxis] * np.fft.fftshift(np.fft.fft2(basis_funct[:,:,:] * X[np.newaxis, :, :], axes=(1, 2)), axes=(1, 2))).reshape(num_coeffs, sizex*sizey)
        coeff_coils[i,:] = np.matmul(np.matmul(Y_i, np.transpose(np.conjugate(A_i))), np.linalg.inv(np.matmul(A_i, np.transpose(np.conjugate(A_i)))))

    return coeff_coils


# Pytorch implementation

def complex_inverse(ctensor, ntry=5) -> "ComplexTensor":
    # Code from https://github.com/kamo-naoyuki/pytorch_complex/blob/b6f82d076f8e6ad035e8573a007c467391d646ff/torch_complex/tensor.py
    # m x n x n
    in_size = ctensor.size()
    a = ctensor.view(-1, ctensor.size(-1), ctensor.size(-1))
    # see "The Matrix Cookbook" (http://www2.imm.dtu.dk/pubdb/p.php?3274)
    # "Section 4.3"
    for i in range(ntry):
        t = i * 0.1

        e = a.real + t * a.imag
        f = a.imag - t * a.real

        try:
            x = torch.matmul(f, e.inverse())
            z = (e + torch.matmul(x, f)).inverse()
        except Exception:
            if i == ntry - 1:
                raise
            continue

        if t != 0.0:
            eye = torch.eye(
                a.real.size(-1), dtype=a.real.dtype, device=a.real.device
            )[None]
            o_real = torch.matmul(z, (eye - t * x))
            o_imag = -torch.matmul(z, (t * eye + x))
        else:
            o_real = z
            o_imag = -torch.matmul(z, x)

        o = torch.complex(o_real, o_imag, device=a.real.device)
        return o.view(*in_size)

def pytorch_UFT(x, uspat, sensmaps):
    # inp: [nx, ny], [nx, ny]
    # out: [nx, ny, ns]
    return uspat[:, :].unsqueeze(0) * fftshift(torch.fft.fftn(sensmaps * x[:, :].unsqueeze(0), dim=(1, 2)), dim=(1, 2))

def pytorch_sense_estimation_ls(Y, X, basis_funct, uspat, device):
    """
    Estimation the sensitivity maps for MRI Reconstruction using polynomial basis functions. Implemented with pytorch and GPU accelerated Least squares solution. 
    :param data: y (undersampled kspace) [c x n x m]
    :param X: predicted reconstruction estimate [c x n x m]
    :param max_basis_order:
    :return coefficients: Least squares coefficients for basis functions
    """

    Y = torch.from_numpy(Y).to(device)
    X = torch.from_numpy(X).to(device)
    basis_funct = torch.from_numpy(basis_funct).to(device)
    uspat = torch.from_numpy(uspat).to(device)

    num_coils, sizex, sizey = y.shape
    num_coeffs = basis_funct.shape[1]

    coeff_coils = torch.zeros((num_coils, num_coeffs), dtype=torch.cfloat, device=y.real.device)
    # XA - Y = 0
    for i in range(num_coils):
        Y = y[i,:,:].reshape(sizex*sizey) 
        A = pytorch_UFT(X, uspat, basis_funct[i,:,:,:]).reshape(num_coeffs, sizex*sizey) 
        coeff = torch.matmul(torch.matmul(Y, torch.transpose(torch.conj(A), 0, 1)), complex_inverse(torch.matmul(A, torch.transpose(torch.conj(A), 0, 1))))
        coeff_coils[i,:] = coeff

    return coeff_coils.detach().cpu().numpy()


















