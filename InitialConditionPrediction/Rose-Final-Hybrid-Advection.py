import torch
import matplotlib.pyplot as plt
import sys

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")

# Parameters
a = 1.0
totalT = 1.0
numberOfTimeSteps = 100
dt = totalT / numberOfTimeSteps
Deltax = 0.02

print(f"a: {a}")
print(f"numberOfTimeSteps: {numberOfTimeSteps}")
print(f"dt: {dt}")
print(f"Deltax: {Deltax}")

# Diagonal matrix A2, shape (100, 100), 1/dt on diagonal
A2 = torch.eye(numberOfTimeSteps, dtype=torch.float64) * (1.0/dt)

# Vectors um and uref: data pasted line-by-line as in MATLAB
um = torch.tensor([0.0000e+00, 1.9817e-31, 1.6564e-29, 6.8627e-28, 1.8793e-26, 3.8264e-25,
        6.1797e-24, 8.2461e-23, 9.3518e-22, 9.2023e-21, 7.9823e-20, 6.1807e-19,
        4.3156e-18, 2.7404e-17, 1.5939e-16, 8.5440e-16, 4.2437e-15, 1.9624e-14,
        8.4840e-14, 3.4426e-13, 1.3157e-12, 4.7519e-12, 1.6266e-11, 5.2929e-11,
        1.6416e-10, 4.8651e-10, 1.3812e-09, 3.7653e-09, 9.8782e-09, 2.4992e-08,
        6.1102e-08, 1.4462e-07, 3.3195e-07, 7.4003e-07, 1.6046e-06, 3.3880e-06,
        6.9729e-06, 1.4001e-05, 2.7448e-05, 5.2565e-05, 9.8384e-05, 1.8003e-04,
        3.2219e-04, 5.6404e-04, 9.6607e-04, 1.6191e-03, 2.6557e-03, 4.2634e-03,
        6.6994e-03, 1.0305e-02, 1.5519e-02, 2.2881e-02, 3.3031e-02, 4.6690e-02,
        6.4628e-02, 8.7604e-02, 1.1630e-01, 1.5120e-01, 1.9255e-01, 2.4016e-01,
        2.9342e-01, 3.5117e-01, 4.1172e-01, 4.7288e-01, 5.3208e-01, 5.8656e-01,
        6.3351e-01, 6.7038e-01, 6.9507e-01, 7.0612e-01, 7.0289e-01, 6.8559e-01,
        6.5527e-01, 6.1369e-01, 5.6321e-01, 5.0650e-01, 4.4636e-01, 3.8548e-01,
        3.2622e-01, 2.7054e-01, 2.1986e-01, 1.7509e-01, 1.3665e-01, 1.0450e-01,
        7.8317e-02, 5.7514e-02, 4.1389e-02, 2.9186e-02, 2.0167e-02, 1.3655e-02,
        9.0592e-03, 5.8890e-03, 3.7509e-03, 2.3407e-03, 1.4312e-03, 8.5730e-04,
        5.0311e-04, 2.8925e-04, 1.6290e-04, 8.9875e-05], dtype=torch.float64)

uref = torch.tensor([0.0000e+00, 7.9120e-38, 8.0830e-36, 4.0933e-34, 1.3700e-32, 3.4097e-31,
        6.7308e-30, 1.0978e-28, 1.5218e-27, 1.8303e-26, 1.9406e-25, 1.8367e-24,
        1.5676e-23, 1.2168e-22, 8.6507e-22, 5.6684e-21, 3.4415e-20, 1.9453e-19,
        1.0281e-18, 5.0999e-18, 2.3828e-17, 1.0520e-16, 4.4026e-16, 1.7514e-15,
        6.6409e-15, 2.4063e-14, 8.3529e-14, 2.7842e-13, 8.9312e-13, 2.7631e-12,
        8.2605e-12, 2.3909e-11, 6.7112e-11, 1.8297e-10, 4.8520e-10, 1.2529e-09,
        3.1538e-09, 7.7452e-09, 1.8571e-08, 4.3497e-08, 9.9575e-08, 2.2286e-07,
        4.8782e-07, 1.0445e-06, 2.1881e-06, 4.4854e-06, 8.9982e-06, 1.7668e-05,
        3.3956e-05, 6.3885e-05, 1.1767e-04, 2.1219e-04, 3.7465e-04, 6.4771e-04,
        1.0965e-03, 1.8179e-03, 2.9517e-03, 4.6937e-03, 7.3103e-03, 1.1152e-02,
        1.6665e-02, 2.4393e-02, 3.4978e-02, 4.9134e-02, 6.7617e-02, 9.1166e-02,
        1.2043e-01, 1.5586e-01, 1.9764e-01, 2.4557e-01, 2.9897e-01, 3.5665e-01,
        4.1691e-01, 4.7754e-01, 5.3601e-01, 5.8956e-01, 6.3545e-01, 6.7116e-01,
        6.9467e-01, 7.0459e-01, 7.0032e-01, 6.8213e-01, 6.5108e-01, 6.0898e-01,
        5.5818e-01, 5.0134e-01, 4.4125e-01, 3.8056e-01, 3.2161e-01, 2.6633e-01,
        2.1610e-01, 1.7181e-01, 1.3384e-01, 1.0215e-01, 7.6390e-02, 5.5966e-02,
        4.0170e-02, 2.8245e-02, 1.9456e-02, 1.3128e-02], dtype=torch.float64)

# Compute delJdelum
delJdelum = 2 * Deltax * (um - uref)
delJdelum = delJdelum.unsqueeze(1)  # shape (100,1) for matrix calc

# Backward sweep: solve lambdam = A2^{-1} @ delJdelum
lambdam = torch.linalg.solve(A2, delJdelum)

# A1 matrix construction
A1 = torch.zeros((numberOfTimeSteps, numberOfTimeSteps), dtype=torch.float64)
A1[torch.arange(numberOfTimeSteps), torch.arange(numberOfTimeSteps)] = -a / Deltax
A1[torch.arange(numberOfTimeSteps-1), torch.arange(1, numberOfTimeSteps)] = (-1/dt) + a/Deltax

# Adjoint time loop (backward)
for i in range(numberOfTimeSteps-1, 0, -1):
    lambdai = -torch.linalg.solve(A2, A1.T @ lambdam)
    lambdam = lambdai

# Final sensitivity: dJdphi = -lambdam^T @ A2
dJdphi = -(lambdam.T @ A2).squeeze()  # shape (100,)

# Visualization 1: Gradient
#plt.figure(1)
#plt.plot(range(1, 101), dJdphi.cpu().numpy(), '-')
#plt.xlabel('Spatial step #')
#plt.ylabel('dJ/dphi')
#plt.title('Gradient dJ/dphi')

# Visualization 2: um and uref
plt.figure(2)
plt.plot(range(1, 101), uref.cpu().numpy(), '-', label='uref')
plt.plot(range(1, 101), um.cpu().numpy(), '-', label='um')
plt.legend()
plt.title('uref vs um')

# Section: urefphi and umphi
urefphi = torch.tensor([ 1.46020529, 1.38217669, 1.3086497, 1.2420592, 1.18410081, 1.13562651,
    1.09667395, 1.06660531, 1.04431304, 1.02844397, 1.0176001, 1.01048937,
    1.00601649, 1.00331856, 1.00175882, 1.00089497, 1.00043686, 1.00020439,
    1.00009157, 1.00003925, 1.00001608, 1.00000629, 1.00000235, 1.00000083,
    1.00000028, 1.00000009, 1.00000003, 1.00000001, 1., 1.,
    1., 1., 1., 1., 1., 1.,
    1., 1., 1., 1., 1., 1.,
    1., 1., 1., 1., 1., 1.00000001,
    1.00000003, 1.00000009, 1.00000028, 1.00000083, 1.00000235, 1.00000629,
    1.00001608, 1.00003925, 1.00009157, 1.00020439, 1.00043686, 1.00089497,
    1.00175882, 1.00331856, 1.00601649, 1.01048937, 1.0176001, 1.02844397,
    1.04431304, 1.06660531, 1.09667395, 1.13562651, 1.18410081, 1.2420592,
    1.3086497, 1.38217669, 1.46020529, 1.53979434, 1.61782245, 1.69134795,
    1.7579345, 1.81588311, 1.86433424, 1.90323448, 1.9331903, 1.9552501,
    1.97066107, 1.98064108, 1.98619207, 1.98796702, 1.98619207, 1.98064108,
    1.97066107, 1.9552501, 1.9331903, 1.90323448, 1.86433424, 1.81588311,
    1.7579345, 1.69134795, 1.61782245, 1.53979434
], dtype=torch.float64)
umphi = torch.tensor([1.65176364, 1.57731391, 1.49999968, 1.42268505, 1.3482338, 1.27919714,
    1.21755724, 1.16456856, 1.12071586, 1.08578235, 1.05899999, 1.03924251,
    1.02522108, 1.01565089, 1.00937046, 1.00540895, 1.00300804, 1.00161049,
    1.00082951, 1.00041073, 1.00019535, 1.00008918, 1.00003904, 1.00001638,
    1.00000658, 1.00000253, 1.00000093, 1.00000032, 1.00000011, 1.00000003,
    1.00000001, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.00000001, 1.00000003, 1.00000011, 1.00000032, 1.00000093,
    1.00000253, 1.00000658, 1.00001638, 1.00003904, 1.00008918, 1.00019535,
    1.00041073, 1.00082951, 1.00161049, 1.00300804, 1.00540895, 1.00937046,
    1.01565089, 1.02522108, 1.03924251, 1.05899999, 1.08578235, 1.12071586,
    1.16456856, 1.21755724, 1.27919714, 1.3482338, 1.42268505, 1.49999968,
    1.57731391, 1.65176364, 1.72079628, 1.78242637, 1.83539239, 1.87919496,
    1.9140223, 1.94058928, 1.95992797, 1.97316843, 1.98134107, 1.98522059,
    1.98522059, 1.98134107, 1.97316843, 1.95992797, 1.94058928, 1.9140223,
    1.87919496, 1.83539239, 1.78242637, 1.72079628
], dtype=torch.float64)

# chickens vector: 2 * Deltax * (umphi - urefphi)
##chickens = 2 * Deltax * (umphi - urefphi)

#plt.figure(3)
#plt.plot(range(1, 101), urefphi.cpu().numpy(), '-', label='urefphi')
#plt.plot(range(1, 101), umphi.cpu().numpy(), '-', label='umphi')
#plt.title('urefphi vs umphi')
#plt.legend()

##plt.figure(4)
#plt.plot(range(1, 101), chickens.cpu().numpy(), '-')
#plt.title('chickens')

# Compute gradient right and left
gradRight = dJdphi
lambdam = torch.linalg.solve(A2, delJdelum)
A1 = torch.zeros((numberOfTimeSteps, numberOfTimeSteps), dtype=torch.float64)
A1[torch.arange(numberOfTimeSteps), torch.arange(numberOfTimeSteps)] = -a / Deltax
A1[torch.arange(numberOfTimeSteps-1), torch.arange(1, numberOfTimeSteps)] = (-1/dt) + a/Deltax
for i in range(numberOfTimeSteps-1, 0, -1):
    lambdai = -torch.linalg.solve(A2, A1 @ lambdam)
    lambdam = lambdai
gradLeft = -(lambdam.T @ A2).squeeze()
fulldJdphi = gradLeft + gradRight

print(f"Printed dJdphi (Rose implementation): {fulldJdphi}")

plt.figure(5)
plt.plot(range(1, 101), fulldJdphi.cpu().numpy(), '-')
plt.title('Full dJdphi')

# Variance computation
#variance = torch.var(torch.stack((chickens, fulldJdphi)), dim=0, unbiased=False)
#averageVariance = torch.mean(variance).item()

#print(f"Average Variance: {averageVariance:.6f}")

plt.show()