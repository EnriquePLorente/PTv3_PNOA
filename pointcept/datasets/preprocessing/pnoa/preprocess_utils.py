import laspy 
import numpy as np

class PNOATilePreprocessing()
    def __init__(self, laz_file_path, output_dir, tile_size, overlap):
        self.laz_file_path = laz_file_path
        self.output_dir = output_dir
        self.tile_size = tile_size
        self.overlap = overlap
        

        self.point_number = None
        self.bbox = None
        self.coord = None
        self.color = None
        self.strength = None
        
        # Crea directorio de salida si no existe
        os.makedirs(self.output_dir, exist_ok=True)

    def read_laz(self):
        with laspy.open(self.las_file_path) as f:
            
            import pdb; pdb.set_trace()
            self.point_number = f.header.point_count
            x_min, y_min, z_min = f.header.mins
            x_max, y_max, z_max = f.header.maxs
            self.bbox = (x_min, x_max, y_min, y_max)

            las = f.read()
            self.coord = np.vstack([las.X, las.Y, las.Z]).T
            self.color = np.vstack([las.red, las.green, las.blue]).T
            self.strength = np.vstack([las.intensity]).T

            return (coord, color, strength, bbox)

        def _save_tile_to_npy(self, tile_name):
            """
            Guarda las características en formato npy dentro de la partición correspondiente
            """
            pass

        def _data_filter_per_tile(self, tile_bbox):
            """
            Debe filtrar los puntos que se encuentran dentro del bbox del tile
            """
            pass

        def _generate_tiles_from_laz(self):
            laz_file_name = laz_file_path[:]
            x_min, x_max, y_min, y_max = laz_bbox 
            
            paso = tile_size - overlap
            x_inicio = np.arange(x_min, x_max, paso)
            y_inicio = np.arange(y_min, y_max, paso)
            d
            tiles_validos = set()
            tile_names = set()
            
            for i, x in enumerate(x_inicio):
                for j, y in enumerate(y_inicio):
                    
                    # 1. Ajusta el inicio de las celdas de los bordes
                    if x + tile_size > x_max:
                        x = x_max - tile_size
                        
                    if y + tile_size > y_max:
                        y = y_max - tile_size
                        
                    # Calcula el final
                    x_fin = x + tile_size
                    y_fin = y + tile_size
                    
                    tile_coords = (x, y, x_fin, y_fin)
                    if tile_coords not in tiles_validos:
                        tiles_validos.add(tile_coords)

                        
                        print(f"Tile generado: X[{x:.2f} a {x_fin:.2f}] | Y[{y:.2f} a {y_fin:.2f}]")

            return tiles_validos

        def preprocess_laz_file(laz_file_path, tile_size, overlap):
            coord, color, strength, laz_bbox = read_laz(las_file_path)
            generate_tiles_from_laz(tile_size, overlap, laz_bbox, laz_file_path)





