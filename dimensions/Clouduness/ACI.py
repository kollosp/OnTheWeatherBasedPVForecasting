from __future__ import annotations
if __name__ == "__main__": import __config__

import numpy as np
from dimensions.BaseDimension import BaseDimension
import pandas as pd
from dimensions.Clouduness.OCI import OCI
from dimensions.Clouduness.VCI import VCI

class ACI(BaseDimension):
    """ACI transformer class. ACI is computed with OCI and VCI. """
    def __init__(self, window_size, latitude_degrees:float, longitude_degrees:float, **kwargs) -> None:
        """

        :param window_size:
        :param latitude_degrees:
        :param longitude_degrees:
        :param kwargs: base_dimensions: List[str,str]. First str is a name of column containing already calculated OCI, while
                       the second str is a name of column containing VCI.
        """
        self.window_size = window_size
        self.latitude_degrees = latitude_degrees
        self.longitude_degrees = longitude_degrees
        self.oci = OCI(window_size=window_size, latitude_degrees = latitude_degrees, longitude_degrees = longitude_degrees)
        self.vci = VCI(window_size=window_size)
        super(ACI, self).__init__(required_dimensions=2, dimension_name=kwargs.pop("dimension_name", "ACI"), **kwargs)

    def fit(self, y: pd.DataFrame | pd.Series, X:pd.DataFrame | pd.Series | None = None):
        x = self.extract_y_X(y,X) # if base_dimensions is correct then x is DataFrame. OCI, VCI in columns
        # fit models if no base_dimensions passed
        if isinstance(x, pd.Series):
            self.oci.fit(y,X)
            self.vci.fit(y,X)

        return self

    def _transform(self, y: pd.DataFrame | pd.Series, X:pd.DataFrame | pd.Series | None = None) -> pd.Series | pd.DataFrame:
        """Class transformes only the y provieded in fit function"""
        x = self.extract_y_X(y,X) # if base_dimensions is correct then x is DataFrame. OCI, VCI in columns
        # create data only if no base_dimensions passed
        if isinstance(x, pd.Series):
            oci = self.oci.transform(y,X).values.flatten() # flat oci (1D np array)
            vci = self.vci.transform(y,X).values.flatten() # flat vci (1D np array)
        else:
            oci = x[self.base_dimensions[0]].values.flatten() # flat oci (1D np array)
            vci = x[self.base_dimensions[1]].values.flatten() # flat vci (1D np array)

        angles = np.arctan2(oci, vci)
        angles = np.degrees(angles)

        angles = (90 - angles) % 360
        angles = np.where(angles > 180, 360 - angles, angles)

        angles[(oci == 0) & (vci == 0)] = 0

        return pd.Series(data=angles, index=y.index)

