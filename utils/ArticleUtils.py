import os
from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from matplotlib.backends.backend_pdf import PdfPages

def print_or_save_figures(figures: List[figure], path=None) -> None:
    """
    Function shows or saves list of figures. If save mode is selected then all figures are stored in .pdf file
    :param figures: matplotlib figures to be save or print
    :param path: a path to store pdf file. If not provided figures are displayed
    :return: None
    """
    if path is not None:
        pp = PdfPages(path)
        for f in figures:
            print(f"Working in {str(f)}...")
            if isinstance(f, list):
                for ff in f:
                    pp.savefig(ff)
            else:
                pp.savefig(f)
        pp.close()
    else:
        for f in figures:
            print(f"Working in {str(f)}...")
            if isinstance(f, list):
                for ff in f:
                    ff.show()
            else:
                f.show()
        plt.show()

def df_2_latex_str(df: pd.DataFrame, caption, command_name, float_format=None) -> str:
    """Function generates latex string representation of pd.DataFrame"""
    txt = "\\newcommand*\\" + command_name + "{" + df.to_latex(
        caption=caption,
        label=f"tab:{command_name}",
        float_format="{:0.1f}".format if float_format is None else float_format,
        na_rep="-"
    ) + "}"
    txt = txt.replace("\\begin{table}", "\\begin{table}\\centering\\small")
    return txt

def concatenate_df(list_dfs:List[List[pd.DataFrame]]):
    return pd.concat(list_dfs)

def join_dataframes(list_dfs:List[List[pd.DataFrame]], indexes:List[str]=None, index_names=None) -> List[pd.DataFrame]:
    """
    Function joins similar dataframes (e.g. representing metrics obtained on several datasets) and creates MultiIndex
    on row-index
    :param index_names:
    :param list_dfs: dataframes to be combined (along row dimension)
    :param indexes: list of higher level index values. One index in indexes for each df in dfs
    :return: combined dataframe
    """
    if indexes is None:
        indexes = np.arrange(len(list_dfs))
    ret = [pd.DataFrame()] * len(list_dfs[0])

    for dfs, indx in zip(list_dfs, indexes):
        for i, d in enumerate(dfs):
            d.index = pd.MultiIndex.from_product([[indx], d.index], names=index_names)
            ret[i] = pd.concat([ret[i],d])
    return ret