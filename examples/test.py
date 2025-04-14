import numpy as np
c = np.arange(5)
a = np.array([0,0,np.nan, 0,0])
b= np.array([0,0,np.nan, np.nan,0])

d = c[~np.isnan(a) & ~np.isnan(b) ]
print(d)