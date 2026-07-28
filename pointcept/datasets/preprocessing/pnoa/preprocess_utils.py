import os
import logging

# log_level = logging.DEBUG
# logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')

def read_txt_distribution(distribution_txt_path):
    logging.debug(f"Leyendo archivo: {distribution_txt_path} ")

    dist_dict = {'train': [], 'val': [], 'test': []}
    current_split = None
    
    with open(distribution_txt_path, 'r') as f:

        lines = f.read().splitlines()
        
        for line in lines:
            line = line.strip()
            
            if not line:
                continue
                
            if line in dist_dict.keys():
                current_split = line
            
            elif current_split is not None:
                dist_dict[current_split].append(line)

    for split in dist_dict:
        dist_dict[split] = tuple(dist_dict[split])

        
    return dist_dict

def create_txt_file(distribution_txt_path):
    logging.debug(f"Creando archivo: {distribution_txt_path}")
    with open(distribution_txt_path, 'w') as f:
        pass


def generate_txt_distribution(data_dir, distribution_txt_path):

    logging.debug(f"Archivo detectado: {distribution_txt_path}")

    create_txt_file(distribution_txt_path)

    splits = ['train','val','test']

    with open(distribution_txt_path, 'a') as f:
        for split in splits:
            split_distribution_dir = os.path.join(data_dir, split)
            logging.debug(split_distribution_dir)

            if os.path.exists(split_distribution_dir):
                files_names = [file for file in os.listdir(split_distribution_dir) if file.lower().endswith('.laz')]
                f.write(split + '\n')

                for file_name in files_names:
                    f.write(file_name[:-4] + ',' + '\n')
            
            else:
                logging.warning(f"La carpeta {split} no existe")
   



