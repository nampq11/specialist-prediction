import pickle

def save_file(name, obj):
    """
    Function to save an object as a pickle file.
    
    Parameters:
        name: The name of the pickle file to save
        obj: The object to saved.
    """
    with open(name, 'wb') as f:
        pickle.dump(obj, f)

def load_file(name):
    """
    Function to load a pickle object.
    
    Parameters:
        name: The name of the pickle file to load.
    
    Returns:
        The loaded object from pickle file.
    """
    return pickle.load(open(name, 'rb'))