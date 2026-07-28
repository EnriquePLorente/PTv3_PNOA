"""
Se utiliza el preproceso base de S3DIS para aprovechar la estructura del código
TODO: El objeto DATASET tieen ya integrado Strength pero habrá que añadir returns
TODO: Comenzar el preproceso a partir del yaml. Único argumento será el path al yaml
"""

#Si en un fututo se quiere añadir normales fijarse en el código base

import os
import argparse
import glob
import numpy as np

try:
    import open3d
except ImportError:
    import warnings

    warnings.warn("Please install open3d for parsing normal")

try:
    import trimesh
except ImportError:
    import warnings

    warnings.warn("Please install trimesh for parsing normal")

from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

import PnoaLaz
import preprocess_utils
import logging



if __name__ == "__main__":
    dir_data_path = "data\\pnoa\\raw"
    distribution_txt_path = "pointcept\\datasets\\preprocessing\\pnoa\\distribucion_datos.txt"
    data_dir_splits = "D:\\eplorente\\data\\Aragon"

    lista_ficheros = os.listdir(dir_data_path)

    PNOAPreprocess = PnoaLaz.PNOALazPreprocessing(
    log_level=logging.DEBUG)

    PNOAPreprocess.load_config(r"configs\\config\\pnoa.yaml")

    for fichero_laz in lista_ficheros:
        PNOAPreprocess.run(fichero_laz)
        pass

    if os.path.exists(distribution_txt_path):
        preprocess_utils.read_txt_distribution(distribution_txt_path)
        logging.warning(f"El fichero {distribution_txt_path} ya existe")
    else:
        preprocess_utils.generate_txt_distribution(data_dir_splits, distribution_txt_path)
        preprocess_utils.read_txt_distribution(distribution_txt_path)

