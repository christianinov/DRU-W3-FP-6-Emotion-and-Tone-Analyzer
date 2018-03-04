import sys
import os
import pandas as pd

categories = {
        'anger':'negative',
        'disgust':'negative',
        'sad':'negative',
        'fear':'negative',
        'boredom':'neutral',
        'neutral':'neutral',
        'calm':'neutral',
        'happy':'positive',
        'surprised':'positive',
        
}

def process_categories_df(df):
    y = [categories[val] for val.lower() in df[0].values]
    return pd.DataFrame(y)


def process_categories(file_from,file_to):
    if not os.path.isfile(file_from):
        raise FileNotFoundError(file_from)
    if not file_from.endswith('.csv'):
        raise ValueError('Csv file needs to be given')
    df = pd.read_csv(file_from, header=None)
    y = [categories[val] for val.lower() in df[0].values]
    df_new  = pd.DataFrame(y)
    df_new.to_csv(file_to)
