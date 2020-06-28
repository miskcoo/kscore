# Nonparametric Score Estimators

Yuhao Zhou, Jiaxin Shi, Jun Zhu. https://arxiv.org/abs/2005.10099

## Toy Example

```
python -m examples.spiral --lam=1.0e-5 --kernel=curlfree_imq --estimator=nu
```

<img src="spiral-gradient.png" width=300 height=300 /><img src="spiral-density.png" width=300 height=300 />

## Dependencies

```
Tensorflow >= 1.14.0
```

## Usage

* Create a score estimator

  ```python
  # Tikhonov regularization (Theorem 3.1), equivalent to KEF (Example 3.5)
  kef_estimator = kscore.estimators.Tikhonov(lam=0.0001, use_cg=False, kernel=kscore.kernels.CurlFreeIMQ)
  
  # Tikhonov regularization + Conjugate Gradient (KEF-CG, Example 3.8)
  kefcg_estimator = kscore.estimators.Tikhonov(lam=0.0001, use_cg=True, kernel=kscore.kernels.CurlFreeIMQ)
  
  # Tikhonov regularization + Nystrom approximation (Appendix C.1), equivalent to NKEF (Example C.1) using 60% samples
  nkef_estimator = kscore.estimators.Tikhonov(lam=0.0001, use_cg=False, subsample_rate=0.6, kernel=kscore.kernels.CurlFreeIMQ)
  
  # Tikhonov regularization + Nystrom approximation + Conjugate Gradient
  nkefcg_estimator = kscore.estimators.Tikhonov(lam=0.0001, use_cg=True, subsample_rate=0.6, kernel=kscore.kernels.CurlFreeIMQ)
  
  # Landweber iteration (Theorem 3.4)
  landweber_estimator = kscore.estimators.Landweber(lam=0.00001, kernel=kscore.kernels.CurlFreeIMQ)
  landweber_estimator = kscore.estimators.Landweber(iternum=100, kernel=kscore.kernels.CurlFreeIMQ)
  
  # nu-method (Example C.4)
  nu_estimator = kscore.estimators.NuMethod(lam=0.00001, kernel=kscore.kernels.CurlFreeIMQ)
  nu_estimator = kscore.estimators.NuMethod(iternum=100, kernel=kscore.kernels.CurlFreeIMQ)
  
  # Spectral cut-off regularization (Theorem 3.2), equivalent to SSGE (Example 3.6) using 90% eigenvalues
  ssge_estimator = kscore.estimators.SpectralCutoff(keep_rate=0.9, kernel=kscore.kernels.DiagonalIMQ)
  
  # Original Stein estimator
  stein_estimator = kscore.estimators.Stein(lam=0.001)
  ```

* Fit the score estimator using samples

  ```python
  # manually specify the hyperparameter
  estimator.fit(samples, kernel_hyperparams=kernel_width)
  # automatically choose the hyperparameter (using the median trick)
  estimator.fit(samples)
  ```

* Predict the score

  ```python
  gradient = estimator.compute_gradients(x)
  ```

* Predict the energy (unnormalized log-density)

  ```python
  log_p = estimator.compute_energy(x)   # only for curl-free kernels
  ```

* Construct other curl-free kernels (see `kscore/kernels/curlfree_gaussian.py`)

