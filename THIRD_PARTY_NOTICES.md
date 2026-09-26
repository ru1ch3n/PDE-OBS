# Third-Party Notices

This repository provides an original benchmark harness. It **does not redistribute** third-party datasets, pretrained weights, or upstream source code. Users must download external assets from official sources and comply with their licenses/terms.

## DiffusionPDE (code / datasets / pretrained checkpoints)
- Official repo:
  ```text
  https://github.com/jhhuangchloe/DiffusionPDE
  ```
- Paper:
  ```text
  https://arxiv.org/abs/2406.17763
  ```
- License/terms: see upstream repository.
- PDE-OBS does not vendor DiffusionPDE source. The numerical protocol cites
  its published use of second-order finite differences, FNO-derived periodic
  solvers, and related PDE-generation settings. DiffusionPDE is CC BY-NC-SA 4.0;
  do not copy its training code or redistribute its assets as MIT content.

## NeuralOperator (FNO ecosystem)
- Official repo:
  ```text
  https://github.com/neuraloperator/neuraloperator
  ```
- Original data-generation reference:
  ```text
  https://github.com/ixScience/fourier_neural_operator/tree/master/data_generation
  ```
- License/terms: see upstream repository.
- The periodic vorticity solver design follows the MIT-licensed FNO
  data-generation reference (pseudospectral streamfunction, 2/3 dealiasing,
  Crank--Nicolson viscosity, and the reference `1e-4` internal time step).
  PDE-OBS contains an independently written, NumPy-based implementation and
  records its own solver ID.

## Convolutional Neural Operator (CNO)
- Official repo:
  ```text
  https://github.com/camlab-ethz/ConvolutionalNeuralOperator
  ```
- Paper:
  ```text
  https://arxiv.org/abs/2302.01178
  ```
- License/terms: see upstream repository.

## DeepONet reference implementation
- Reference repo:
  ```text
  https://github.com/lululxvi/deeponet
  ```
- License/terms: see upstream repository.
- The campaign adapter is independently implemented from the published
  branch--trunk construction. No DeepONet source is redistributed. The
  inspected reference revision is recorded in the campaign registry and is
  licensed CC BY-NC-SA 4.0 upstream.

## Transolver
- Official repo:
  ```text
  https://github.com/thuml/Transolver
  ```
- Paper:
  ```text
  https://arxiv.org/abs/2402.02366
  ```
- License/terms: MIT in the inspected upstream revision.
- PDE-OBS uses an independently integrated structured-grid, mask-conditioned
  physics-attention adapter and does not vendor the upstream source tree.

## GNOT
- Official repo:
  ```text
  https://github.com/thu-ml/GNOT
  ```
- Paper:
  ```text
  https://proceedings.mlr.press/v202/hao23c.html
  ```
- No license file was found in the inspected upstream revision. Consequently,
  PDE-OBS does not copy or redistribute GNOT source. Its local adapter is an
  independent paper-guided implementation of heterogeneous normalized
  attention and is not described as an exact code reproduction.

## PINNs reference implementation
- Reference repo:
  ```text
  https://github.com/maziarraissi/PINNs
  ```
- License/terms: see upstream repository.
