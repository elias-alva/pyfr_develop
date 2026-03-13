from collections import namedtuple
import numpy as np
import os

from pyfr.mpiutil import get_comm_rank_root, mpi
from pyfr.plugins.base import BaseSolverPlugin
from pyfr.points import PointLocator, PointSampler
from pyfr.quadrules import get_quadrule
from pyfr.plugins.sampler import _process_con_to_pri

ALMInfo = namedtuple('alminfo', ['nloc', 'forc'])

class ALMPlugin(BaseSolverPlugin):
    name = 'alm'
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
                'pyfr.plugins.kernels.alm', 'alm',
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

        # List of points to be sampled and format
        pts = self.cfg.getliteral(cfgsect, 'samp-pts')
        self.pts = np.array(pts)

        # Alm parameters
        e = self.cfg.getfloat(cfgsect, 'e')
        omega = self.cfg.getfloat(cfgsect, 'omega')
        Amp = self.cfg.getfloat(cfgsect, 'Amp')
        self.M = self.cfg.getfloat(cfgsect,'M') 
        self.c = self.cfg.getfloat(cfgsect, 'chord')

        # Define the future positions
        self.h = lambda t: Amp*np.cos(omega*t) + pts[0][1]
        self.hdot = lambda t: (-1)*Amp*omega*np.sin(omega*t)

        # Relavent macro parameters
        self.macro_params = {'eph': e, 'Amp': Amp, 'eph2': e*e}


    def __call__(self, intg):
        # For debug
        comm, rank, root = get_comm_rank_root()

        # Updates force and locations
        self._update_forc(intg)
        pts, forc = self.pts, self.forc 

        for etype, info in self.alm_info.items():
            # Updates
            info.nloc.set(pts)
            info.forc.set(forc)

        # Renew time
        self.told = intg.tcurr



    def _update_solution(self, intg): #where is the point, what is the solution associated to this
        # New location
        self.pts[0][1] = self.h(intg.tcurr)

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

    def _update_forc(self, intg): 
        #comm, rank, root = get_comm_rank_root()

        # Update to new location and solution
        nsoln = self._update_solution(intg)
        
        # hh position
        hh = self.h(intg.tcurr)

        # V new
        vh = self.hdot(intg.tcurr)

        # V DNS, temp here since only one point matters
        rho, Udns, Vdns = nsoln[0, :-1]

        # AoA
        alpha = np.arctan2((Vdns - vh),Udns)

        # Relative velocity square
        Vrel2 = Udns**2 + (vh - Vdns)**2

        forcx = np.pi*rho*Vrel2*self.c*alpha*np.sin(alpha)/np.sqrt(1-self.M*self.M) ########################################## cambiado
        forcy = np.pi*rho*Vrel2*self.c*alpha*np.cos(alpha)/np.sqrt(1-self.M*self.M) ########################################## cambiado
        fqs = np.pi*rho*self.c*Vrel2*(-vh)/Udns/np.sqrt(1-self.M*self.M)

        if forcx:
            with open('force_file2.txt','a') as f:
                line_to_write = f"{intg.tcurr:.4f} {forcx:.8f} {forcy:.8f} {fqs:.8f} {hh:.8f} {vh:.8f} {alpha:.8f} {rho:.8f} {Vdns:.8f} {Udns:.8f}\n"
                f.write(line_to_write)

        # temporary: Create a forcing with the shape of pts
        # and fill the forcing terms 
        self.forc[0, 0] = (-1)*forcx
        self.forc[0, 1] = (-1)*forcy

        
