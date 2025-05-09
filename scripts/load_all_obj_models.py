import os

def load_all_obj_models(model_directory):
    model_files = []
    for filename in os.listdir(model_directory):
        if filename.endswith('.obj'):
            model_files.append(os.path.join(model_directory, filename))
    return model_files