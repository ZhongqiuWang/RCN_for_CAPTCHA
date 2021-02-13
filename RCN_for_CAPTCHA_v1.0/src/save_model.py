import shelve


def save_model_to_local(model_factors, model_dir):
    file = shelve.open(model_dir)
    data_key = "my_data"
    file[data_key] = model_factors
    file.close()


def load_model_from_local(model_dir):
    file = shelve.open(model_dir)
    data_key = "my_data"
    model_factors = file[data_key]
    file.close()
    return model_factors
