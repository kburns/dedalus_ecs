"""
dt(X) = L(X) + N(X,X)

Stationary state:
    L(X) + N(X,X) = 0

Newton's method:
    L(X1) + N(X0,X1) + N(X1,X0) = N(X0,X0)
    A(X1) = B(X0)

Preconditioned:
    Linv A (X1) = Linv B(X0)

    Linv A(X1) = X1 + Y1
    L(Y1) = N(X0,X1) + N(X1,X0)

    Linv B(X0) = Z0
    L(Z0) = N(X0, X0)

"""

import numpy as np
from mpi4py import MPI
import time
import scipy.sparse.linalg as spla
import h5py

import dedalus.dev as dev
from dedalus import public as de
from dedalus.extras import flow_tools

import logging
logger = logging.getLogger(__name__)


# Parameters
Nx = 32
Ny = 32
Nz = 32
Lx = 5.511566058929462
Ly = 2.5132741228718345
Lz = 2
Re = 400
solver_tolerance = 1e-4
max_iter = 10
newton_tolerance = 1e-16

# Bases and domain
x_basis = de.Fourier('x', Nx, interval=(0, Lx), dealias=3/2)
y_basis = de.Fourier('y', Ny, interval=(0, Ly), dealias=3/2)
z_basis = de.Chebyshev('z', Nz, interval=(-Lz/2, Lz/2), dealias=3/2)
domain = de.Domain([x_basis, y_basis, z_basis], grid_dtype=np.float64)

# Fields
field_names = ['p','u','v','w','uz','vz','wz']
X0 = dev.system.FieldSystem(field_names, domain)
X1 = dev.system.FieldSystem(field_names, domain)

# Define Y1 problem
problem = de.LBVP(domain, variables=field_names)
problem.parameters['Re'] = Re
for name in field_names:
    problem.parameters[name+'0'] = X0[name]
    problem.parameters[name+'1'] = X1[name]
problem.substitutions['U'] = "z"
problem.substitutions['L(a,az)'] = "dx(dx(a)) + dy(dy(a)) + dz(az)"
problem.add_equation("dx(u) + dy(v) + wz = 0")
problem.add_equation("(1/Re)*L(u,uz) - dx(p) - U*dx(u) - w*dz(U) = -(dx(u1*u0) + dx(u1*u0) + dy(v1*u0) + dy(u1*v0) + dz(w1*u0) + dz(u1*w0))")
problem.add_equation("(1/Re)*L(v,vz) - dy(p) - U*dx(v)           = -(dx(u1*v0) + dx(v1*u0) + dy(v1*v0) + dy(v1*v0) + dz(w1*v0) + dz(v1*w0))")
problem.add_equation("(1/Re)*L(w,wz) - dz(p) - U*dx(w)           = -(dx(u1*w0) + dx(w1*u0) + dy(v1*w0) + dy(w1*v0) + dz(w1*w0) + dz(w1*w0))")
problem.add_equation("uz - dz(u) = 0")
problem.add_equation("vz - dz(v) = 0")
problem.add_equation("wz - dz(w) = 0")
problem.add_bc("left(u) = 0")
problem.add_bc("left(v) = 0")
problem.add_bc("left(w) = 0")
problem.add_bc("right(u) = 0")
problem.add_bc("right(v) = 0")
problem.add_bc("right(w) = 0", condition="(nx != 0)")
problem.add_bc("right(p) = 0", condition="(nx == 0)")
Y1_solver = problem.build_solver()
Y1 = Y1_solver.state

# Define Z0 problem
problem = de.LBVP(domain, variables=field_names)
problem.parameters['Re'] = Re
for name in field_names:
    problem.parameters[name+'0'] = X0[name]
problem.substitutions['U'] = "z"
problem.substitutions['L(a,az)'] = "dx(dx(a)) + dy(dy(a)) + dz(az)"
problem.add_equation("dx(u) + dy(v) + wz = 0")
problem.add_equation("(1/Re)*L(u,uz) - dx(p) - U*dx(u) - w*dz(U) = -(dx(u0*u0) + dy(v0*u0) + dz(w0*u0))")
problem.add_equation("(1/Re)*L(v,vz) - dy(p) - U*dx(v)           = -(dx(u0*v0) + dy(v0*v0) + dz(w0*v0))")
problem.add_equation("(1/Re)*L(w,wz) - dz(p) - U*dx(w)           = -(dx(u0*w0) + dy(v0*w0) + dz(w0*w0))")
problem.add_equation("uz - dz(u) = 0")
problem.add_equation("vz - dz(v) = 0")
problem.add_equation("wz - dz(w) = 0")
problem.add_bc("left(u) = 0")
problem.add_bc("left(v) = 0")
problem.add_bc("left(w) = 0")
problem.add_bc("right(u) = 0")
problem.add_bc("right(v) = 0")
problem.add_bc("right(w) = 0", condition="(nx != 0)")
problem.add_bc("right(p) = 0", condition="(nx == 0)")
Z0_solver = problem.build_solver()
Z0 = Z0_solver.state

# Wrappers
array_shape = X0.data.shape
array_size = np.prod(array_shape)

def Linv_A(X1_array):
    # Set X1 data
    set_state_data(X1, X1_array)
    # Compute Y1
    Y1_solver.solve()
    # Return X1 + Y1
    return state_to_array(X1) + state_to_array(Y1)

def Linv_B():
    # Compute Z0
    Z0_solver.solve()
    # Return Z0
    return state_to_array(Z0)

def set_state_data(state, array):
    state.data[:] = array.reshape(array_shape)
    state.scatter()

def state_to_array(state):
    state.gather()
    return state.data.ravel()

shape = (array_size, array_size)
A = spla.LinearOperator(shape, matvec=Linv_A)

def callback(rk):
    print('GMRES residual:', np.linalg.norm(rk))

def newton_iteration():
    b = Linv_B()
    X1_array, info = spla.gmres(A, b, x0=state_to_array(X0), tol=solver_tolerance, callback=callback)
    set_state_data(X1, X1_array)

# Initial conditions
x, y, z = domain.grids()
for name in field_names:
    X0[name].set_scales(1)
with h5py.File('data/eq1.h5', mode='r') as file:
    X0['u']['g'] = (file['data/u'][0].transpose((0, 2, 1))[:,:,:-1] + file['data/u'][0].transpose((0, 2, 1))[:,:,1:]) / 2
    X0['v']['g'] = (file['data/u'][2].transpose((0, 2, 1))[:,:,:-1] + file['data/u'][2].transpose((0, 2, 1))[:,:,1:] )/ 2
    X0['w']['g'] = (file['data/u'][1].transpose((0, 2, 1))[:,:,:-1] + file['data/u'][1].transpose((0, 2, 1))[:,:,1:]) / 2

# Newton iteration
pert_norm = np.inf
iter = 0
while (iter < max_iter) and (pert_norm > newton_tolerance):
    # Compute X1
    newton_iteration()
    # Print progress
    X0_array = state_to_array(X0)
    X1_array = state_to_array(X1)
    pert_norm = np.linalg.norm(X1_array-X0_array)
    new_norm = np.linalg.norm(X1_array)
    print()
    print('Pert norm:', pert_norm)
    print('New norm:', new_norm)
    # Update solution
    set_state_data(X0, X1_array)
    iter += 1

