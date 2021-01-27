import argparse
from utils import generate_US_pattern, rmse
import numpy as np
import h5py
import sigpy as sp
import sigpy.mri as mr
from sense_estimation import create_basis_functions, sense_estimation_ls
import pickle

if __name__ == "__main__":
  parser = argparse.ArgumentParser(prog='PROG')
  parser.add_argument('--path', type=str, default='data/testdata.h5', help='Full path to h5 file for reconstruction') 
  parser.add_argument('--slice', type=int, default=0, help='Slice to reconstruct, 0 = all slices (default: 0)')
  parser.add_argument('--usfact', type=int, default=4, help='Undersampling factor (default: 4)')
  parser.add_argument('--basis_order', type=int, default=8, help='Polynomial basis max order (default: 8)')
  parser.add_argument('--num_iter', type=int, default=30, help='Number of iterations (default: 50)')

  args = parser.parse_args()
  path = args.path
  sli = args.slice
  R = args.usfact
  max_basis_order = args.basis_order
  num_iter = args.num_iter

  print('______JSENSE RECONSTRUCTION______')
  print('PATH: ', path, '   SLICE: ', sli, '    Under sample factor: ', R, '    Iterations: ', num_iter)

  ## OPEN DATA 
  with h5py.File(path, 'r') as fdset:
    ksp_sli = fdset['kspace'][:]  # The shape of kspace tensor is (number of slices, number of coils, height, width)
    mask = fdset['mask'][:]

  img_sli = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(ksp_sli, axes=(1,2)), axes=(1,2)), axes=(1,2)) # Change to normal view
  gt_img = np.sqrt(np.sum(np.square(np.abs(img_sli.copy())), axis=0))  # Save rss ground truth 
  ksp_sli = np.fft.fftshift(np.fft.fft2(img_sli, axes=(1,2)), axes=(1,2)) # Create kspace data
  num_coils, rows, cols = ksp_sli.shape 
  
  # Create and apply US pattern
  us_pat, num_low_freqs = generate_US_pattern(ksp_sli.shape, R=R) 
  us_ksp = us_pat * ksp_sli

  # Estimate initial sensemaps using Espirit 
  est_sensemap = np.fft.fftshift(mr.app.EspiritCalib(us_ksp, 
    calib_width=num_low_freqs, thresh=0.02, kernel_width=6, crop=0.01, max_iter=100, 
    show_pbar=False, device=0).run().get(),axes=(1, 2))

  # Initializing image reconstruction
  recs = np.zeros((num_iter, rows, cols), dtype=complex)
  img_sli_rec = np.fft.ifft2(np.fft.ifftshift(us_ksp, axes=(1, 2)), axes=(1, 2))
  recs[0] = np.sum(img_sli_rec * np.conjugate(est_sensemap), axis=0) / np.sum(est_sensemap * np.conjugate(est_sensemap), axis=0) 

  # Sense Reconstruction for evaluation 
  print('SENSE RECONSTRUCTION')
  for i in range(num_iter-1):
    # Data consistency projection
    # ===============================================
    rec_ksp = (1 - us_pat) * np.fft.fftshift(np.fft.fft2(est_sensemap * recs[i, np.newaxis, :, :], axes=(1, 2)), axes=(1, 2))
    rec_usksp = rec_ksp + us_ksp

    img_sli_rec = np.fft.ifft2(np.fft.ifftshift(rec_usksp, axes=(1, 2)), axes=(1, 2))
    recs[i+1] = np.sum(img_sli_rec * np.conjugate(est_sensemap), axis=0) / np.sum(est_sensemap * np.conjugate(est_sensemap), axis=0)
    print("Iteration: %(i)s         RMSE score: %(rmse)s" % {'i': i, 'rmse': rmse(mask * np.sqrt(np.sum(np.square(np.abs(img_sli_rec)), axis=0)), gt_img)})
  print('<<DONE>>')
  print('Reconstruction saved as: rec_SENSE')

  pickle.dump(recs[-1], open('rec_SENSE', 'wb'))

  # Initializing sense reconstruction
  num_coeffs = (max_basis_order + 1) ** 2
  basis_funct = create_basis_functions(rows, cols, max_basis_order,show_plot=False)
  coeffs_array = sense_estimation_ls(us_ksp, recs[0], basis_funct, us_pat)

  est_sensemap = np.sum(coeffs_array[:, :, np.newaxis, np.newaxis] * basis_funct[np.newaxis], 1)
  
  # First step of image reconstruction
  img_sli_rec = np.fft.ifft2(np.fft.ifftshift(us_ksp, axes=(1, 2)), axes=(1, 2))
  recs[1] = np.sum(img_sli_rec * np.conjugate(est_sensemap), axis=0) / np.sum(est_sensemap * np.conjugate(est_sensemap), axis=0) 

  print('JSENSE RECONSTRUCTION')
  for i in range(1, num_iter-1):
    # Sense reconstruction 
    # ===============================================
    coeffs_array = sense_estimation_ls(us_ksp, recs[i], basis_funct, us_pat)

    # Data consistency projection
    # ===============================================
    rec_ksp = (1 - us_pat) * np.fft.fftshift(np.fft.fft2(est_sensemap * recs[i, np.newaxis, :, :], axes=(1, 2)), axes=(1, 2))
    rec_usksp = rec_ksp + us_ksp

    # Update sensemap  
    est_sensemap = np.sum(coeffs_array[:, :, np.newaxis, np.newaxis] * basis_funct[np.newaxis], 1)

    # Create next reconstruction
    img_sli_rec = np.fft.ifft2(np.fft.ifftshift(rec_usksp, axes=(1, 2)), axes=(1, 2))
    recs[i+1] = np.sum(img_sli_rec * np.conjugate(est_sensemap), axis=0) / np.sum(est_sensemap * np.conjugate(est_sensemap), axis=0) 
        
    print("Iteration: %(i)s         RMSE score: %(rmse)s" % {'i': i, 'rmse': rmse(mask * np.sqrt(np.sum(np.square(np.abs(img_sli_rec)), axis=0)), gt_img)})

  print('<<DONE>>')
  print('Reconstruction saved as: recon_JSENSE')

  pickle.dump(recs[-1], open('rec_JSENSE', 'wb'))
