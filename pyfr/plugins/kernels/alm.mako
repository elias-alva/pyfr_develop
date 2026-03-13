<% import numpy as np %>

<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:macro name='alm' params='t, u, ploc, src' externs='nloc, forc'>
  fpdtype_t b = 1/${eph}/${eph}/${np.pi} * exp(-((ploc[0] - nloc[0][0])*(ploc[0] - nloc[0][0]) + (ploc[1] - nloc[0][1])*(ploc[1] - nloc[0][1]))*${eph2});
  % for i in range(ndims):
    src[${i + 1}] += b*forc[0][${i}];
  % endfor
  % for i in range(ndims):
    src[${nvars - 1}] += b*forc[0][${i}]*u[${i+1}];
  % endfor
</%pyfr:macro>

