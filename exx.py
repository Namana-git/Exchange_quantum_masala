import numpy as np
from qtm.constants import RYDBERG, ELECTRONVOLT
from qtm.lattice import RealLattice
from qtm.crystal import BasisAtoms, Crystal
from qtm.pseudo import UPFv2Data
from qtm.kpts import gen_monkhorst_pack_grid
from qtm.gspace import GSpace
from qtm.mpi import QTMComm
from qtm.dft import DFTCommMod, scf
from qtm.dft.kswfn import KSWfn
from qtm.constants import FPI

from qtm.io_utils.dft_printers import print_scf_status

from qtm import qtmconfig
from qtm.logger import qtmlogger
qtmconfig.fft_backend = 'mkl_fft'

from mpi4py.MPI import COMM_WORLD
comm_world = QTMComm(COMM_WORLD)
dftcomm = DFTCommMod(comm_world, 1)

# Lattice
reallat = RealLattice.from_alat(alat=10.2,  # Bohr
                                a1=[-0.5,  0. ,  0.5],
                                a2=[ 0. ,  0.5,  0.5],
                                a3=[-0.5,  0.5,  0. ])

# Atom Basis
si_oncv = UPFv2Data.from_file('Si_ONCV_PBE-1.2.upf')
si_atoms = BasisAtoms('si', si_oncv, 28.086, reallat, np.array(
    [[0.875, 0.875, 0.875], [0.125, 0.125, 0.125]]
).T)

crystal = Crystal(reallat, [si_atoms, ])  # Represents the crystal

#print(crystal.recilat)
# Generating k-points from a Monkhorst Pack grid (reduced to the crystal's IBZ)
mpgrid_shape = (4, 4, 4)
mpgrid_shift = (False, False, False)
kpts = gen_monkhorst_pack_grid(crystal, mpgrid_shape, mpgrid_shift, False, False)
#print(kpts.k_cart)
#print(kpts.k_cart)


# -----Setting up G-Space of calculation-----
ecut_wfn = 10 * RYDBERG
# NOTE: In future version, hard grid (charge/pot) and smooth-grid (wavefun)
# can be set independently
ecut_rho = 4 * ecut_wfn
grho = GSpace(crystal.recilat, ecut_rho)
gwfn = grho

numbnd = 4  # Ensure adequate # of bands if system is not an insulator
conv_thr = 1E-8 * RYDBERG
diago_thr_init = 1E-2 * RYDBERG




#print(l_wfn_kgrp)
#l_wfn_kgrp[k1][0].
def enr(l_wfn_kgrp):
     cell_vol = l_wfn_kgrp[0][0].gkspc.gwfn.reallat_cellvol
     #print(cell_vol)
     nkpts = kpts.numkpts #no of kpoints
     #print(nkpts)
     kpoints = kpts.k_cryst #array containing kpoints of shape nkpts*3
     k_weights = kpts.numkpts
     nocc = len(l_wfn_kgrp[0][0].occ)
     #print(nocc)    
     
     sum = 0
      #fft grid size
     for k1 in range(nkpts):
        for k2 in range(nkpts):
            phi_all_n_k1 =   l_wfn_kgrp[k1][0].evc_gk.to_r()
            phi_all_n_k2 =   l_wfn_kgrp[k2][0].evc_gk.to_r()
            #print(phi_all_n_k1)
            #print(np.shape(phi_all_n_k1))
            #print(k1,k2)
            for i in range(int(nocc)):
                for j in range(int(nocc)):
                    #print(k1,k2,i,j)
                    #print(nocc)  
                    phi_i_k1 =  phi_all_n_k1[i]  #band i
                    #print(np.shape(phi_i_k1._data))
                    phi_j_k2 =  phi_all_n_k2[j]  #band j

                    Yofr = phi_i_k1.conj().copy() 
                    Yofr._data *= phi_j_k2._data 
                    YofG = Yofr.to_g()
        
                    YofG._data /= np.prod(l_wfn_kgrp[0][0].gkspc.gwfn.grid_shape)    

                    recip_cart = np.asarray(crystal.recilat.axes_cart)
                    k1_pt = kpts.k_cryst[0,k1]*recip_cart[0,:] + kpts.k_cryst[1,k1]*recip_cart[1,:] + kpts.k_cryst[2,k1]*recip_cart[2,:] 
                    k2_pt = kpts.k_cryst[0,k2]*recip_cart[0,:] + kpts.k_cryst[1,k2]*recip_cart[1,:] + kpts.k_cryst[2,k2]*recip_cart[2,:] 
                    ng = np.shape(YofG._data)
                    #print(ng)
                    #print(np.shape(k1_pt))
                    k1_min_k2 = np.tile((k1_pt-k2_pt),(len(ng),1)).T
                    #print(np.shape(k1_min_k2))
                    #print(YofG.gspc.g_cart)
                    #print(np.shape(YofG.gspc.g_cart)) #(3,ng)
                    G_min_k1_min_k2 = YofG.gspc.g_cart - k1_min_k2 #(3,ng)
                    sqr_G_min_k1_min_k2 = np.sum((np.multiply(np.conjugate(G_min_k1_min_k2),G_min_k1_min_k2)),axis=0)
                    #print(np.shape(sqr_G_min_k1_min_k2))
                    #print(sqr_G_min_k1_min_k2)
                    #print(sqr_G_min_k1_min_k2)
                    #print(YofG.gspc.g_norm2)
                    #print(YofG.gspc.g_cart)

                    val = 0
                    if(k1==k2):
                       val =  np.sum(abs(YofG.data[1:]) ** 2 / (sqr_G_min_k1_min_k2[1:]))
                    else:
                        val =  np.sum(abs(YofG.data[:]) ** 2 / (sqr_G_min_k1_min_k2[:]))
                    sum += val
                    #if (i==j):
                    #print(val)

                    
     print(sum)
     #print(nkpts)
     
     exx_energy = 4*np.pi*sum/((nkpts*nkpts*cell_vol)) 
     return(exx_energy)

def compute(
    l_wfn_kgrp_outer: list[WavefunGType], l_wfn_kgrp_inner: list[WavefunGType]
) -> list[WavefunGType]:
    """Generate exact exchange potential and energy, given a list of wavefunctions.

    E_x = -1/4 * 4pi/V_c \sum_{k,n} \sum{k'!=k,m} f'_{k,n} f'_{k',m} \sum_{G}

    Parameters
    ----------
    l_wfn_kgrp : list[list[KSWfn]]
        list of KS wavefunctions

    Returns
    -------
    l_vexx_psi: list of exact exchange applied to vectors (in G-space)
    l_en_exx: list of exact exchange energies
    """
    # FIXME: Very bad assumption
    # FIXME: We also need 2 sets of occupancy
    occ = [1, 1, 1, 1] + [0] * (len(l_wfn_kgrp_outer) - 4)

    l_Vxphi = []

    Vs = l_wfn_kgrp_outer[0].gkspc.gwfn.reallat_cellvol
    # Rc = (3 * Vs / (FPI)) ** (1 / 3)
    s = 0
    one = 1 - (1e-5)

    # For now, only ik=0 point. Later convert these ik's to loop over two different k-point indices.
    ik = 0
    kgrp_i = [wfn.to_r() for wfn in l_wfn_kgrp_outer][ik]
    kgrp_j = [wfn.to_r() for wfn in l_wfn_kgrp_inner][ik]

    grid_size = np.prod(l_wfn_kgrp_outer[0][0].gkspc.gwfn.grid_shape)

    # Loop over bands, because we are only computing gamma point
    for i in range(len(l_wfn_kgrp_outer)):
        sum = kgrp_i[0].copy()
        sum._data *= 0.0
        for j in range(len(l_wfn_kgrp_inner)):
            if occ[j] < one:
                continue

            phi_i = kgrp_i[i]
            phi_j = kgrp_j[j]

            Yofr = phi_j.conj().copy()
            Yofr._data *= phi_i._data
            Yofr._data /= grid_size

            YofG = Yofr.to_g()

            # NOTE: Check whether the zeroth element of YofG.data corresponds to G=0.
            YofG.data[1:] = (
                -FPI
                * YofG.data[1:]
                / (YofG.gspc.g_norm2[1:])
            )
            YofG._data[0] = 0.0  # -FPI * 0.5 * YofG._data[0] * Rc**2

            YofG_r = YofG.to_r()
            YofG_r /= grid_size
            # YofG_r /= Vs
            sum._data += YofG_r.data * phi_j.data 

        l_Vxphi.append(sum.to_g())

    return l_Vxphi

out = scf(dftcomm, crystal, kpts, grho, gwfn,
          numbnd, is_spin=False, is_noncolin=False,
          symm_rho=False, rho_start=None, occ_typ='fixed',
          conv_thr=conv_thr, diago_thr_init=diago_thr_init,
          iter_printer=print_scf_status)

scf_converged, rho, l_wfn_kgrp, en = out
ener = enr(l_wfn_kgrp)
#ener1 = enr1(l_wfn_kgrp)
#print(ener1)
print(ener)
print("SCF Routine has exited")
print(qtmlogger)
