from collections import namedtuple
import numpy as np
import os

# from pyfr.plugins.coefficients import SD7003clcd as CLCD
from pyfr.plugins.coefficients import CLCD13x6 as C13x6
from pyfr.plugins.coefficients import betas13x6 as betas
from pyfr.mpiutil import get_comm_rank_root, mpi
from pyfr.plugins.base import BaseSolverPlugin
from pyfr.points import PointLocator, PointSampler
from pyfr.quadrules import get_quadrule
from pyfr.plugins.sampler import _process_con_to_pri

ALMInfo = namedtuple('alminfo', ['nloc', 'forc'])

class ALMPlugindev3_v2(BaseSolverPlugin):
    name = 'almdev3_v2'
    systems = ['*']
    formulations = ['dual', 'std']
    dimensions = [2, 3]

    def __init__(self, intg, cfgsect):
        super().__init__(intg, cfgsect)

        # Underlying elements class
        self.elementscls = intg.system.elementscls

        # Also element maps
        self.ele_map = intg.system.ele_map

        # Initialize the parameters
        self._init_param(cfgsect)
        
        # force files
        self.force_files = [f"force_point_{i:03d}.txt" for i in range(len(self.pts))]
        for fname in self.force_files:
            with open(fname, 'w') as f:
                f.write("# t Fx Fy Fz alpha rho Udns Vdns Ux Uy Px Py\n")

        # Data format for ALM calculations
        self._process = _process_con_to_pri(self.elementscls, self.ndims,
                                                self.cfg)
        #self._process = None


        # Construct and configure the point sampler and locator
        self.mesh = intg.system.mesh
        self.psampler = PointSampler(self.mesh, self.pts)
        self.ubases = self._config_ubases(intg)
        #self.psampler.configure_with_intg_nvars(intg, self.nvars)

        self.locf = ['cidx', 'eidx', 'tloc']
        self.plocator = PointLocator(self.mesh)


        # allocate the inital time
        self.told = intg.tcurr

        # Data type
        self.dtype = intg.system.backend.fpdtype

        # Prepare ALM kernels
        self.alm_info = self._alm_kern(intg)

    def _config_ubases(self, intg):
        # Get the solution bases from the system
        ubases = {etype: eles.basis.ubasis
                  for etype, eles in intg.system.ele_map.items()}
        
        return ubases

    def _alm_kern(self, intg):
        pts = self.pts
        # Initialize data to broadcast
        self.forc = forc = np.zeros(pts.shape, self.dtype)
        nploc = np.zeros(pts.shape, self.dtype)

        npts, ndim = len(pts), self.ndims

        # Add macro kernel to backends
        alm_info = {}
        for etype, eles in self.ele_map.items():
            eles.add_src_macro(
                'pyfr.plugins.kernels.almdev3', 'almdev3',
                self.macro_params, ploc=True, soln=False
            )

            alm_info[etype] = vs = ALMInfo(
                    intg.backend.matrix(pts.shape, pts, tags={'align'}),
                    intg.backend.matrix(forc.shape, forc, tags={'align'}),
                    )

            eles._set_external('forc', f'in broadcast fpdtype_t[{npts}][{ndim}]', value=vs.forc)
            eles._set_external('nloc', f'in broadcast fpdtype_t[{npts}][{ndim}]', value=vs.nloc)

            #eles._set_external('info', f'in broadcast fpdtype_t[2][2]', value=vs.info)
        return alm_info

    def _init_param(self, cfgsect):

        # Alm parameters
        e = self.cfg.getfloat(cfgsect, 'e')
        omega = self.cfg.getfloat(cfgsect, 'omega')
        self.M = self.cfg.getfloat(cfgsect,'M')
        self.mu = self.cfg.getfloat(cfgsect,'mu')
        # r_scalar = self.cfg.getfloat(cfgsect, 'r')
        # th0 = self.cfg.getliteral(cfgsect, 'thet0')
        r_sample = np.array(self.cfg.getliteral(cfgsect, 'r'))
        theta_blades = np.array(self.cfg.getliteral(cfgsect, 'thet0'))
        
        # self.c = self.cfg.getfloat(cfgsect, 'chord')
        n_blades = len(theta_blades)
        n_pts_per_blade = len(r_sample)
        npts = n_blades * n_pts_per_blade

        # r and theta arrays per point
        r = np.tile(r_sample, n_blades)
        thetas0 = np.repeat(theta_blades, n_pts_per_blade)

        self.betas, self.c, self.dr = betas.pitch(r)

        # List of points to be sampled and format
        # pts = self.cfg.getliteral(cfgsect, 'samp-pts')
        x = np.zeros_like(r)
        y = r * np.cos(np.radians(thetas0))
        z = r * np.sin(np.radians(thetas0))
        self.pts = np.column_stack((x, y, z))

        self.thetas = [(lambda t, th0=th: omega*t + np.radians(th0)) for th in thetas0]
        self.xx = 0.0
        self.yy = [lambda t, r0=r0, th0=th0: r0*np.cos(omega*t + np.radians(th0))
               for r0, th0 in zip(r, thetas0)]
        self.zz = [lambda t, r0=r0, th0=th0: r0*np.sin(omega*t + np.radians(th0))
               for r0, th0 in zip(r, thetas0)]
        self.omegar = omega*r

        self.macro_params = {'eph': e,'eph2': 1/e**2, 'ephpi3': 1/e**3/(np.pi)**(3/2), 'r': r,'npts':npts}


    def __call__(self, intg):
        # For debug
        comm, rank, root = get_comm_rank_root()

        # Update force
        self._update_forc(intg)
        pts, forc = self.pts, self.forc 

        for etype, info in self.alm_info.items():
            # Updates
            info.nloc.set(pts)
            info.forc.set(forc)

        # Renew time
        self.told = intg.tcurr
        
        # Update locations for body forces in N-S
        self._advance_positions(intg)



    def _update_solution(self, intg): #where is the point, what is the solution associated to this
        # New location
        # self.pts[0][1] = self.h(intg.tcurr)
        # for i in range(len(self.pts)):
        #     self.pts[i][1] = self.yy[i](intg.tcurr)
        #     self.pts[i][2] = self.zz[i](intg.tcurr)

        # Locate the new point list
        locs = self.plocator.locate(self.pts)[self.locf]

        # Fetch the solution
        soln = list(intg.soln)

        # Sample the solution
        self.psampler.locs = locs 
        self.psampler._configure_ubases_nvars(self.ubases, self.nvars)
        samps = self.psampler.sample(soln, process=self._process)

        # Broadcast to all ranks
        comm, rank, root = get_comm_rank_root()
        samps = comm.bcast(samps, root = root)

        return samps
    
    def _advance_positions(self, intg):
        for i, th in enumerate(self.thetas):
            theta_new = th(intg.tcurr + intg._dt)
            r = np.linalg.norm(self.pts[i][1:3])
            self.pts[i][1] = r * np.cos(theta_new)
            self.pts[i][2] = r * np.sin(theta_new)
            # self.pts[i][0] = 0 # rotor plane

    def _update_forc(self, intg): 
        #comm, rank, root = get_comm_rank_root()

        # Update to new location and solution
        nsoln = self._update_solution(intg)
        
        # hh position
        Py = [yy(intg.tcurr) for yy in self.yy]
        Pz = [zz(intg.tcurr) for zz in self.zz]

        # V new
        thetan = np.array([thetas(intg.tcurr) for thetas in self.thetas])

        # V DNS, temp here since only one point matters
        # rho, Udns, Vdns = nsoln[0, :-1]
        nsoln = np.array(nsoln)
        rho   = nsoln[:, 0] # [ρ0, ρ1, ρ2, ...]
        Udns  = nsoln[:, 1]/nsoln[:,0] # [U0, U1, U2, ...]
        Vdns  = nsoln[:, 2]/nsoln[:,0] # [V0, V1, V2, ...]
        Wdns  = nsoln[:, 3]/nsoln[:,0] # [V0, V1, V2, ...]

        # relative AoA(phi) and effective AoA(alpha)
        phi = np.arctan2((Udns),(self.omegar + Vdns*np.sin(thetan) -Wdns*np.cos(thetan)))
        alpha = self.betas - phi

        # Relative velocity square
        Vrel2 = (Udns)**2 + (self.omegar + Vdns*np.sin(thetan) -Wdns*np.cos(thetan))**2
        Vrel = np.sqrt(Vrel2)
        re_l = self.c*rho*Vrel/self.mu

        # Aerodyanmic coefficients and forces
        # Cl = 2*np.pi*(alpha)/np.sqrt(1-self.M*self.M)
        Cl_int, Cd_int = C13x6.clcd13x6(alpha*180/np.pi,re_l)
        Cl = Cl_int/np.sqrt(1-Vrel2*self.M*self.M)
        Cd = Cd_int/np.sqrt(1-Vrel2*self.M*self.M)
        L = 0.5*rho*Vrel2**2*self.c*Cl*self.dr
        D = 0.5*rho*Vrel2**2*self.c*Cd*self.dr  #------> Vrel = (rho*u)**2

        # Actuator line forces
        Fetheta = D* np.cos(phi) + L*np.sin(phi)
        Fx = L*np.cos(phi) - D*np.sin(phi)
        Fy = Fetheta*np.sin(thetan)
        Fz = -Fetheta*np.cos(thetan)
        # forcx = np.pi*rho*Vrel*self.c*alpha*np.sin(alpha)/np.sqrt(1-self.M*self.M) ########################################## cambiado
        # forcy = np.pi*rho*Vrel*self.c*alpha*np.cos(alpha)/np.sqrt(1-self.M*self.M) ########################################## cambiado

        #fqs = np.pi*self.c*(-vh)/np.sqrt(1-self.M*self.M)

        # if forcx:
        #     with open('force_file2.txt','a') as f:
        #         line_to_write = f"{intg.tcurr:.4f} {forcx:.8f} {forcy:.8f} {fqs:.8f} {hh:.8f} {vh:.8f} {alpha:.8f} {rho:.8f} {Vdns:.8f} {Udns:.8f}\n"
        #         f.write(line_to_write)

        # temporary: Create a forcing with the shape of pts
        # and fill the forcing terms 
        self.forc[:,0] = -Fx
        self.forc[:,1] = -Fy
        self.forc[:,2] = -Fz
        
        if intg.nacptsteps % 5 == 0:
            for i in range(len(self.pts)):
                with open(self.force_files[i], 'a') as f:
                    line_to_write = (
                        f"{intg.tcurr:.6f} "
                        f"{Fx[i]:.8e} {Fy[i]:.8e} {Fz[i]:.8e} "
                        f"{alpha[i]:.8e} "
                        f"{rho[i]:.8e} {Udns[i]:.8e} {Vdns[i]:.8e} {Wdns[i]:.8e} "
                        # f"{uu[i]:.8e} {vv[i]:.8e} "
                        f"{re_l[i]:.8e} {self.c[i]:.8e} "
                        f"{Py[i]:.8e} {Pz[i]:.8e}\n"
                    )
                    f.write(line_to_write)
        
