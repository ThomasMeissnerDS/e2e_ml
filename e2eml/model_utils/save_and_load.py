import gc
import pickle


def save_to_production(class_instance, file_name='automl_instance', file_type='.dat', clean=True):
    """
    Takes a pretrained model and saves it via pickle.
    :param class_instance: Takes instance of a class.
    :param file_name: Takes a string containing the whole file name.
    :param file_type: Takes the expected type of file to export.
    :return:
    """
    if clean:
        del class_instance.df_dict
        del class_instance.dataframe
        _ = gc.collect()
    else:
        pass
    filehandler = open(file_name+file_type, 'wb')
    pickle.dump(class_instance, filehandler)
    filehandler.close()


def load_for_production(file_name='automl_instance', file_type='.dat'):
    """
    Load in a pretrained auto ml model. This function will try to load the model as provided.
    It has a fallback logic to impute .dat as file_type in case the import fails initially.
    :param file_name:
    :param file_type:
    :return:
    """
    try:
        filehandler = open(file_name, 'rb')
    except Exception:
        filehandler = open(file_name+file_type, 'rb')
    automl_model = pickle.load(filehandler)
    filehandler.close()
    return automl_model