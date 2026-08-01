import logging
import os
from pathlib import Path

import laspy
import numpy as np
import yaml


class PNOALazPreprocessing:
    """
    Preprocesamiento de ficheros LAZ del PNOA para Pointcept.

    Las clasificaciones originales del LAZ se convierten en:

        0 .. num_classes - 1  -> clases utilizadas para entrenar
        -1                    -> clase ignorada

    Las clases que no aparecen en class_mapping se convierten automáticamente
    en ignore_index=-1.
    """

    IGNORE_INDEX = -1

    def __init__(self, log_level=logging.INFO):
        logging.basicConfig(
            level=log_level,
            format="%(levelname)s: %(message)s",
        )

        # Configuración
        self.laz_dir = None
        self.output_dir = None
        self.tile_size = None
        self.overlap = None
        self.num_classes = None

        # Mapeo de clases
        self.map_classes = None
        self.inverse_map_classes = None
        self.class_names = None

        # Información del LAZ
        self.point_number = None
        self.laz_bbox = None
        self.laz_name = None
        self.laz_file_path = None

        # Datos
        self.coord = None
        self.color = None
        self.strength = None
        self.segment = None

        # Información del tile actual
        self.tile_name = None
        self.tile_output_dir = None

    def load_config(self, config_path):
        """
        Carga la configuración YAML y construye el mapeo:

            clasificación original LAZ -> clase de entrenamiento 0..N-1
        """
        config_path = Path(config_path)

        if not config_path.is_file():
            raise FileNotFoundError(
                f"No se encuentra el fichero de configuración: {config_path}"
            )

        with config_path.open("r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)

        required_sections = {
            "model",
            "class_mapping",
            "dataset",
            "paths",
        }

        missing_sections = required_sections.difference(config)

        if missing_sections:
            raise KeyError(
                "Faltan secciones en el YAML: "
                f"{sorted(missing_sections)}"
            )

        model_config = config["model"]
        class_mapping_config = config["class_mapping"]
        dataset_config = config["dataset"]
        paths_config = config["paths"]

        self.num_classes = int(model_config["num_classes"])
        self.tile_size = float(dataset_config["tile_size"])
        self.overlap = float(dataset_config["overlap"])

        self.output_dir = os.path.normpath(
            paths_config["processed_dataset_dir"]
        )
        self.laz_dir = os.path.normpath(
            paths_config["raw_dataset_dir"]
        )

        if self.num_classes <= 0:
            raise ValueError(
                "model.num_classes debe ser mayor que cero"
            )

        if self.tile_size <= 0:
            raise ValueError(
                "dataset.tile_size debe ser mayor que cero"
            )

        if self.overlap < 0:
            raise ValueError(
                "dataset.overlap no puede ser negativo"
            )

        if self.overlap >= self.tile_size:
            raise ValueError(
                "dataset.overlap debe ser menor que dataset.tile_size"
            )

        if len(class_mapping_config) != self.num_classes:
            raise ValueError(
                f"class_mapping contiene "
                f"{len(class_mapping_config)} clases, "
                f"pero model.num_classes={self.num_classes}"
            )

        self.map_classes = {}
        self.class_names = {}

        # PyYAML conserva el orden declarado en el YAML.
        # Si una clase contiene train_id, se utiliza explícitamente.
        # De lo contrario, se asignan 0, 1, ..., num_classes-1.
        for default_train_id, (class_name, class_info) in enumerate(
            class_mapping_config.items()
        ):
            if "original_ids" not in class_info:
                raise KeyError(
                    f"La clase '{class_name}' no contiene original_ids"
                )

            train_id = int(
                class_info.get("train_id", default_train_id)
            )

            if train_id < 0 or train_id >= self.num_classes:
                raise ValueError(
                    f"La clase '{class_name}' tiene train_id={train_id}. "
                    f"Debe estar entre 0 y {self.num_classes - 1}"
                )

            if train_id in self.map_classes:
                raise ValueError(
                    f"El train_id={train_id} está repetido"
                )

            original_ids = [
                int(raw_id)
                for raw_id in class_info["original_ids"]
            ]

            if not original_ids:
                raise ValueError(
                    f"La clase '{class_name}' no contiene códigos originales"
                )

            self.map_classes[train_id] = original_ids
            self.class_names[train_id] = class_name

        expected_train_ids = set(range(self.num_classes))
        actual_train_ids = set(self.map_classes)

        if actual_train_ids != expected_train_ids:
            raise ValueError(
                "Los train_id deben cubrir exactamente el intervalo "
                f"0..{self.num_classes - 1}. "
                f"Encontrados: {sorted(actual_train_ids)}"
            )

        original_ids = [
            raw_id
            for raw_ids in self.map_classes.values()
            for raw_id in raw_ids
        ]

        if len(original_ids) != len(set(original_ids)):
            duplicated_ids = sorted(
                {
                    raw_id
                    for raw_id in original_ids
                    if original_ids.count(raw_id) > 1
                }
            )

            raise ValueError(
                "Hay códigos originales repetidos en class_mapping: "
                f"{duplicated_ids}"
            )

        self.inverse_map_classes = {
            raw_id: train_id
            for train_id, raw_ids in self.map_classes.items()
            for raw_id in raw_ids
        }

        if not os.path.isdir(self.laz_dir):
            raise FileNotFoundError(
                f"No existe el directorio de LAZ: {self.laz_dir}"
            )

        os.makedirs(self.output_dir, exist_ok=True)

        logging.info(
            "Configuración cargada: num_classes=%d, tile_size=%s, "
            "overlap=%s",
            self.num_classes,
            self.tile_size,
            self.overlap,
        )

        logging.info(
            "Mapeo original -> entrenamiento: %s",
            self.inverse_map_classes,
        )

    def _map_classes(self):
        """
        Convierte las clasificaciones originales del LAZ a clases de
        entrenamiento.

        Cualquier clasificación que no aparezca en el YAML se convierte
        en IGNORE_INDEX=-1.
        """
        if self.segment is None:
            raise RuntimeError(
                "No se han cargado las clasificaciones del LAZ"
            )

        if self.inverse_map_classes is None:
            raise RuntimeError(
                "El mapeo no está inicializado. Ejecuta load_config() primero"
            )

        original_shape = self.segment.shape

        raw_segment = np.asarray(
            self.segment,
            dtype=np.int64,
        ).reshape(-1)

        mapped_segment = np.full(
            raw_segment.shape,
            fill_value=self.IGNORE_INDEX,
            dtype=np.int64,
        )

        for raw_id, train_id in self.inverse_map_classes.items():
            mapped_segment[raw_segment == raw_id] = train_id

        unknown_mask = mapped_segment == self.IGNORE_INDEX

        if np.any(unknown_mask):
            unknown_ids, unknown_counts = np.unique(
                raw_segment[unknown_mask],
                return_counts=True,
            )

            unknown_distribution = {
                int(raw_id): int(count)
                for raw_id, count in zip(
                    unknown_ids,
                    unknown_counts,
                )
            }

            logging.warning(
                "Clasificaciones originales no configuradas; "
                "se guardarán como ignore_index=%d: %s",
                self.IGNORE_INDEX,
                unknown_distribution,
            )

        valid_mask = mapped_segment != self.IGNORE_INDEX

        invalid_mask = valid_mask & (
            (mapped_segment < 0)
            | (mapped_segment >= self.num_classes)
        )

        if np.any(invalid_mask):
            invalid_values = np.unique(
                mapped_segment[invalid_mask]
            )

            raise ValueError(
                "El remapeo produjo etiquetas inválidas: "
                f"{invalid_values.tolist()}"
            )

        self.segment = mapped_segment.reshape(original_shape)

        mapped_ids, mapped_counts = np.unique(
            self.segment,
            return_counts=True,
        )

        mapped_distribution = {
            int(class_id): int(count)
            for class_id, count in zip(
                mapped_ids,
                mapped_counts,
            )
        }

        logging.info(
            "Distribución después del mapeo: %s",
            mapped_distribution,
        )

    def _read_laz(self, laz_file_name):
        """
        Lee un fichero LAZ y extrae coordenadas, color, intensidad
        y clasificación.
        """
        if self.laz_dir is None:
            raise RuntimeError(
                "Primero debes ejecutar load_config()"
            )

        self.laz_name = Path(laz_file_name).stem
        self.laz_file_path = os.path.join(
            self.laz_dir,
            laz_file_name,
        )

        if not os.path.isfile(self.laz_file_path):
            raise FileNotFoundError(
                f"No se encuentra el fichero LAZ: "
                f"{self.laz_file_path}"
            )

        logging.info("Procesando: %s", self.laz_name)

        with laspy.open(self.laz_file_path) as laz_reader:
            self.point_number = laz_reader.header.point_count

            x_min, y_min, _ = laz_reader.header.mins
            x_max, y_max, _ = laz_reader.header.maxs

            self.laz_bbox = (
                float(x_min),
                float(x_max),
                float(y_min),
                float(y_max),
            )

            las = laz_reader.read()

            self.coord = np.column_stack(
                (las.x, las.y, las.z)
            ).astype(np.float64, copy=False)

            self.color = np.column_stack(
                (las.red, las.green, las.blue)
            )

            self.strength = np.asarray(
                las.intensity
            ).reshape(-1, 1)

            # int64 es necesario porque las clases ignoradas utilizan -1.
            self.segment = np.asarray(
                las.classification,
                dtype=np.int64,
            ).reshape(-1, 1)

        expected_points = self.coord.shape[0]

        if not (
            self.color.shape[0]
            == self.strength.shape[0]
            == self.segment.shape[0]
            == expected_points
        ):
            raise ValueError(
                "Las características del LAZ no tienen el mismo "
                "número de puntos. "
                f"Coord={self.coord.shape[0]}, "
                f"Color={self.color.shape[0]}, "
                f"Strength={self.strength.shape[0]}, "
                f"Segment={self.segment.shape[0]}"
            )

        self._map_classes()

    def _save_tile_to_npy(self, **features):
        """
        Guarda las características del tile en archivos NPY.
        """
        if not features:
            raise ValueError(
                "No se han proporcionado características para guardar"
            )

        point_counts = {
            feature_name: feature_data.shape[0]
            for feature_name, feature_data in features.items()
        }

        if len(set(point_counts.values())) != 1:
            raise ValueError(
                "Las características del tile no tienen el mismo "
                f"número de puntos: {point_counts}"
            )

        os.makedirs(self.tile_output_dir, exist_ok=True)

        for feature_name, feature_data in features.items():
            tile_file_path = os.path.join(
                self.tile_output_dir,
                f"{feature_name}.npy",
            )

            np.save(
                tile_file_path,
                feature_data,
                allow_pickle=False,
            )

    def _data_filter_per_tile(self, tile_bbox):
        """
        Selecciona los puntos que están dentro del bounding box del tile.
        """
        x_min, y_min, x_max, y_max = tile_bbox

        mask = (
            (self.coord[:, 0] >= x_min)
            & (self.coord[:, 0] < x_max)
            & (self.coord[:, 1] >= y_min)
            & (self.coord[:, 1] < y_max)
        )

        tile_coord = self.coord[mask]
        tile_color = self.color[mask]
        tile_strength = self.strength[mask]
        tile_segment = self.segment[mask]

        point_counts = {
            "coord": tile_coord.shape[0],
            "color": tile_color.shape[0],
            "strength": tile_strength.shape[0],
            "segment": tile_segment.shape[0],
        }

        if len(set(point_counts.values())) != 1:
            raise ValueError(
                f"Error filtrando el tile {tile_bbox}: "
                f"{point_counts}"
            )

        return (
            tile_coord,
            tile_color,
            tile_strength,
            tile_segment,
        )

    def _validate_tile_segment(self, tile_segment):
        """
        Comprueba que un tile contiene únicamente -1 o clases 0..N-1.
        """
        segment = np.asarray(
            tile_segment,
            dtype=np.int64,
        ).reshape(-1)

        invalid_mask = (
            (segment != self.IGNORE_INDEX)
            & (
                (segment < 0)
                | (segment >= self.num_classes)
            )
        )

        if np.any(invalid_mask):
            invalid_values = np.unique(
                segment[invalid_mask]
            )

            raise ValueError(
                f"El tile {self.tile_name} contiene etiquetas "
                f"inválidas: {invalid_values.tolist()}"
            )

    @staticmethod
    def _calculate_axis_starts(axis_min, axis_max, tile_size, step):
        """
        Calcula los comienzos de los tiles para un eje, ajustando el
        último tile al borde de la nube.
        """
        extent = axis_max - axis_min

        if extent <= 0:
            raise ValueError(
                f"Extensión espacial inválida: {axis_min}, {axis_max}"
            )

        if extent <= tile_size:
            return [float(axis_min)]

        starts = np.arange(
            axis_min,
            axis_max,
            step,
            dtype=np.float64,
        )

        adjusted_starts = []

        for start in starts:
            adjusted_start = min(
                float(start),
                float(axis_max - tile_size),
            )

            if not adjusted_starts or not np.isclose(
                adjusted_start,
                adjusted_starts[-1],
            ):
                adjusted_starts.append(adjusted_start)

        return adjusted_starts

    def _generate_tiles_from_laz(self):
        """
        Divide la nube completa en tiles y guarda sus características.
        """
        x_min, x_max, y_min, y_max = self.laz_bbox

        step = self.tile_size - self.overlap

        x_starts = self._calculate_axis_starts(
            x_min,
            x_max,
            self.tile_size,
            step,
        )

        y_starts = self._calculate_axis_starts(
            y_min,
            y_max,
            self.tile_size,
            step,
        )

        valid_tiles = set()
        saved_tiles = 0
        empty_tiles = 0
        ignored_tiles = 0

        for i, tile_x_min in enumerate(x_starts):
            for j, tile_y_min in enumerate(y_starts):
                tile_x_max = tile_x_min + self.tile_size
                tile_y_max = tile_y_min + self.tile_size

                tile_bbox = (
                    tile_x_min,
                    tile_y_min,
                    tile_x_max,
                    tile_y_max,
                )

                # Redondeo para evitar duplicados por precisión float.
                tile_key = tuple(
                    round(value, 6)
                    for value in tile_bbox
                )

                if tile_key in valid_tiles:
                    continue

                valid_tiles.add(tile_key)

                self.tile_name = (
                    f"{self.laz_name}_tile_{i}_{j}"
                )

                self.tile_output_dir = os.path.join(
                    self.output_dir,
                    self.laz_name,
                    self.tile_name,
                )

                (
                    tile_coord,
                    tile_color,
                    tile_strength,
                    tile_segment,
                ) = self._data_filter_per_tile(tile_bbox)

                if tile_coord.shape[0] == 0:
                    empty_tiles += 1

                    logging.warning(
                        "Tile vacío, no se guardará: %s",
                        self.tile_name,
                    )
                    continue

                self._validate_tile_segment(tile_segment)

                # Un tile completamente ignorado no aporta supervisión.
                if np.all(
                    tile_segment == self.IGNORE_INDEX
                ):
                    ignored_tiles += 1

                    logging.warning(
                        "Tile sin puntos etiquetados, no se guardará: %s",
                        self.tile_name,
                    )
                    continue

                self._save_tile_to_npy(
                    coord=tile_coord,
                    color=tile_color,
                    strength=tile_strength,
                    segment=tile_segment,
                )

                saved_tiles += 1

        logging.info(
            "LAZ %s finalizado: %d tiles guardados, "
            "%d vacíos y %d completamente ignorados",
            self.laz_name,
            saved_tiles,
            empty_tiles,
            ignored_tiles,
        )

        return valid_tiles

    def run(self, laz_file_name):
        """
        Ejecuta el preprocesamiento completo para un fichero LAZ.
        """
        if self.inverse_map_classes is None:
            raise RuntimeError(
                "Debes ejecutar load_config() antes de run()"
            )

        self._read_laz(laz_file_name)
        return self._generate_tiles_from_laz()