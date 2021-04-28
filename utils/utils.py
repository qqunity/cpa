import yaml


def get_config(config_path):
    with open(config_path, mode='r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config


def generate_file_name_with_postfix(file, postfix):
    file_name = '.'.join(file.split('.')[:-1])
    file_extension = file.split('.')[-1]
    return file_name + '_' + postfix + '.' + file_extension
