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

import preprocess_utils
import logging
import yaml



def parse_classes(yaml_path="configs\\pnoa.yaml"):
    

    TODO: Se deben mapear las clases desde 0 hasta n

    classes = [
        "Suelo",
        "Vegetacion_Baja",
        "Vegetacion_Med_Alta",
        "Edificios",
        "Agua",
        "Cables",
        "Torres",
        "Puentes",
        "Vehiculos",
        "Paneles",
        "Aerogeneradores",
    ]
    import pdb; pdb.set_trace()
    class2label = {cls: i for i, cls in enumerate(classes)}
    


def main_process():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--splits",
        required=True,
        nargs="+",
        choices=["muestra_1_677-4615_500x500_urban_train_v2", 
        "muestra_2_678-4615_1000x500_urban_train_v2", 
        "muestra_3_679-4615_500x500_urban_train_v2", 
        "muestra_4_678-4614_500x500_urban_train_v2", 
        "muestra_5_680-4615_500x500_urban_train", 
        "muestra_6_678-4613_500x500_urban_train_v2",
        "muestra_7_677-4612_500x500_urban_train",
        "muestra_8_676-4612_1000x500_urban_train_v2",
        "muestra_9_675-4614_500x500_urban_train_v2",
        "muestra_10_676-4611_500x500_urban_train"],
        
        help="Splits need to process pnoa laz.",
    )
    parser.add_argument(
        "--dataset_root", required=True, help="Ruta a los datos brutos laz (pnoa) -> data\pnoa\raw"
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Ruta donde los datos procesados se guardarán -> data\pnoa\processed",
    )

    parser.add_argument(
        "--tile_size",
        required=True,
        help="Tamaño del tile -> 120",
    )

    parser.add_argument(
        "--overlap",
        required=True,
        help="Tamaño del solape -> 10",
    )

    parser.add_argument(
        "--logging",
        defualt=logging.DEBUG,
        help="Cantidad de información recibida",
    )
    
    parser.add_argument(
        "--num_workers", default=1, type=int, help="Num workers for preprocessing."
    )

    args = parser.parse_args()


if __name__ == "__main__":

    
    PNOAPreprocess = preprocess_utils.PNOALazPreprocessing(laz_file_path="data\\pnoa\\raw\\muestra_1_677-4615_500x500_urban_train_v2.laz", 
    output_dir="data\\pnoa\\processed", 
    tile_size=120, 
    overlap=10,
    log_level=logging.DEBUG)

    PNOAPreprocess.run()
    sys.exit()
    main_process()
