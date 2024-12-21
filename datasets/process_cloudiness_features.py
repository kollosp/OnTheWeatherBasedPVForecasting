if __name__ == "__main__":
    import __config__

from sktimeSEAPF.Optimized import Optimized
from utils.Solar import elevation

import numpy as np
from utils.Plotter import Plotter
from CloudinessFeatures.VCI import Model as VCI
from CloudinessFeatures.OCI import Model as OCI
from datetime import date
from matplotlib import pyplot as plt
import matplotlib.dates as md
import os
import pandas as pd

from SEAIPPF.Model import Model as SEAIPPFModel
from SEAIPPF.Transformers.TransformerTest import TransformerTest



if __name__ == "__main__":
    installation=0
    file_path = "/".join(os.path.abspath(__file__).split("/")[:-1] + ["dataset.csv"])
    df = pd.read_csv(file_path, low_memory=False)
    # self.full_data = self.full_data[30:]
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.index = df['timestamp']
    df.drop(columns=["timestamp"], inplace=True)
    # print_full(df[["0_Power"]].head(288))
    latitude_degrees = df[f"{installation}_Latitude"][0]
    longitude_degrees = df[f"{installation}_Longitude"][0]
    # print_full(df.index.astype('datetime64[s]'))
    timestamps = df.index.astype('datetime64[s]').astype('int')
    df[f"{installation}_elevation"] = elevation(Optimized.from_timestamps(timestamps), latitude_degrees,
                                longitude_degrees) * 180 / np.pi
    df[f"{installation}_NormalizedPower"].fillna(0, inplace=True)
    vci = VCI(window_size=6)
    df[f"{installation}_VCI"] = vci.fit(df[f"{installation}_NormalizedPower"]).transform(df[f"{installation}_NormalizedPower"])

    oci = OCI(model_factory=lambda : SEAIPPFModel(
            latitude_degrees=latitude_degrees,
            longitude_degrees=longitude_degrees,
            x_bins=30,
            y_bins=30,
            bandwidth=0.05,
            interpolation=True,
            enable_debug_params=True,
            transformer=TransformerTest(
                regressor_degrees=12,
                iterative_regressor_degrees=12,
                conv2D_shape_factor=0.05,
                iterations=3,
                stretch=True
            )
        ), window_size=12*3)

    df[f"{installation}_OCI"] = oci.fit(df[f"{installation}_NormalizedPower"]).transform(df[f"{installation}_NormalizedPower"])
    df[f"{installation}_OCI_Model"] = oci.expected_from_model(df[f"{installation}_NormalizedPower"])

    df = pd.DataFrame({
        "NormalizedPower": df[f"{installation}_NormalizedPower"],
        "VCI": df[f"{installation}_VCI"],
        "OCI": df[f"{installation}_OCI"],
        "FingerprintModel": df[f"{installation}_OCI_Model"],
        "Elevation": df[f"{installation}_elevation"]
    })

    file_out = os.path.join("/".join(file_path.split("/")[:-1]), f"weather_dataset_{installation}.csv")
    df.to_csv(file_out)

    plotter = Plotter(df.index, [df[col] for col in df.columns if not col in ["Elevation"]], debug=False)
    plotter.show()
    plt.show()