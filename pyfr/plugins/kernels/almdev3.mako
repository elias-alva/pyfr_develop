<% import numpy as np %>

<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:macro name='almdev3' params='t, u, ploc, src' externs='nloc, forc'>

  fpdtype_t g = 0.0;
  fpdtype_t f[${ndims}] = {};
  fpdtype_t invrho = 1/u[0];


  % for i in range(npts):

    g = exp(-((ploc[0] - nloc[${i}][0])*(ploc[0] - nloc[${i}][0]) + (ploc[1] - nloc[${i}][1])*(ploc[1] - nloc[${i}][1]) + (ploc[2] - nloc[${i}][2])*(ploc[2] - nloc[${i}][2]))*${eph2}) ;
     
    % for j in range(ndims):
      f[${j}] += g*forc[${i}][${j}];
    % endfor 

  % endfor

  // Momentum
  % for i in range(ndims):
    src[${i + 1}] += f[${i}] * ${ephpi3};
  % endfor

  // Energy
  % for i in range(ndims):
    src[${nvars - 1}] += f[${i}]*(u[${i+1}] * invrho ) * ${ephpi3};
  % endfor
</%pyfr:macro>
