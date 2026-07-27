import laspy 
import numpy as np
from pathlib import Path
import os
import logging
import yaml 

class PNOALazPreprocessing():
    def __init__(self, log_level):
        logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')

        self.laz_dir = None
        self.output_dir = None
        self.tile_size = None
        self.overlap = None
        

        self.point_number = None
        self.laz_bbox = None
        self.coord = None
        self.color = None
        self.strength = None
        self.segment = None

        self.tile_name = None
   
        

    def load_config(self, config_path):

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            model_config = config['model']
            class_mapping_config = config['class_mapping']
            dataset_config = config['dataset']
            paths_config = config['paths']

            list_values = [item['original_ids'] for item in class_mapping_config.values()]    #Guarda una lista con los valores de las clases

            self.map_classes = {i:valores for i, valores in enumerate(list_values, start=1)}  #Mapea los valores en un enumerate
            self.inverse_map_classes = {viejo_id: nueva_clase for nueva_clase, viejos_ids in self.map_classes.items() for viejo_id in viejos_ids}

            self.output_dir = paths_config['processed_dataset_dir']
            self.laz_dir = paths_config['raw_dataset_dir']

            self.overlap = dataset_config['overlap']
            self.tile_size = dataset_config['tile_size']

            os.makedirs(self.output_dir, exist_ok=True)

            

    def _map_classes(self):
        lookup_table = np.arange(256, dtype=self.segment.dtype)

        for viejo, nuevo in self.inverse_map_classes.items():
            lookup_table[viejo] = nuevo
        
        self.segment = lookup_table[self.segment]


    def _read_laz(self, laz_file_name):
        """
        Lee el fichero PNOA y extrae la información
        """
        self.laz_name = laz_file_name[:-4]
        logging.info(f"Procesando: {self.laz_name}")
        self.laz_file_path = os.path.join(self.laz_dir, laz_file_name)

        with laspy.open(self.laz_file_path) as f:
            
            self.point_number = f.header.point_count
            x_min, y_min, z_min = f.header.mins
            x_max, y_max, z_max = f.header.maxs
            self.laz_bbox = (x_min, x_max, y_min, y_max)

            las = f.read()  
            self.coord = np.vstack([las.x, las.y, las.z]).T
            self.color = np.vstack([las.red, las.green, las.blue]).T
            self.strength = np.vstack([las.intensity]).T
            self.segment = np.vstack([las.classification]).T
            # Mapea la clasificación con el diccionario
            self._map_classes()


    def _save_tile_to_npy(self, **features):
        """
        Guarda las características en formato npy dentro del directorio del tile.
        Admite cualquier cantidad de características (coord, color, strength, normals, labels...).
        """
        # features es un diccionario. Ejemplo: {'coord': array, 'color': array}
        for feature_name, feature_data in features.items():
            
            tile_file_path = os.path.join(self.tile_output_dir, f"{feature_name}.npy")
            
            np.save(tile_file_path, feature_data)



    def _data_filter_per_tile(self, tile_bbox):
        """
        Filtra los puntos que se encuentran dentro del bbox del tile
        """
        x_min, y_min, x_max, y_max = tile_bbox
        
        # Filtramos con la máscara
        mask = (self.coord[:, 0] >= x_min) & (self.coord[:, 0] < x_max) & \
               (self.coord[:, 1] >= y_min) & (self.coord[:, 1] < y_max) 
               
        tile_coord = self.coord[mask]
        tile_color = self.color[mask]
        tile_strength = self.strength[mask]
        tile_segment = self.segment[mask]

        if not (tile_coord.shape[0] == tile_color.shape[0] == tile_strength.shape[0] == tile_segment.shape[0]):  

            raise ValueError(
                f"Error en el filtrado de tile {tile_bbox}. "
                f"Coord: {tile_coord.shape[0]}, Color: {tile_color.shape[0]}, Strength: {tile_strength.shape[0]}, Classification: {tile_segment.shape[0]}"
            )

        return tile_coord, tile_color, tile_strength, tile_segment
        

    def _generate_tiles_from_laz(self):

        x_min, x_max, y_min, y_max = self.laz_bbox 
        
        paso = self.tile_size - self.overlap
        x_inicio = np.arange(x_min, x_max, paso)
        y_inicio = np.arange(y_min, y_max, paso)
        
        tiles_validos = set()
        tile_names = set()
        
        for i, x in enumerate(x_inicio):
            for j, y in enumerate(y_inicio):
                
                # 1. Ajusta el inicio de las celdas de los bordes
                if x + self.tile_size > x_max:
                    x = x_max - self.tile_size
                    
                if y + self.tile_size > y_max:
                    y = y_max - self.tile_size
                    
                # Calcula el final
                x_fin = x + self.tile_size
                y_fin = y + self.tile_size
                
                tile_coords = (x, y, x_fin, y_fin)
                #Comprueba que no hay duplicados
                if tile_coords not in tiles_validos:
                    tiles_validos.add(tile_coords)

                    self.tile_name = self.laz_name + f"_tile_{i}_{j}"
                    self.tile_output_dir = os.path.join(self.output_dir,self.laz_name,self.tile_name)

                    os.makedirs(self.tile_output_dir, exist_ok=True)

                    tile_coord, tile_color, tile_strength, tile_segment = self._data_filter_per_tile(tile_coords)
                    self._save_tile_to_npy(
                        coord=tile_coord, 
                        color=tile_color, 
                        strength=tile_strength,
                        segment=tile_segment

                    )

        return tiles_validos

    def run(self, laz_file_name):
        self._read_laz(laz_file_name)
        self._generate_tiles_from_laz()





