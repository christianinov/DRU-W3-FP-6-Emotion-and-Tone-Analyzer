import sys
import os
import pandas as pd


def process_categories(file_from=None,file_to=None,df_from=None):
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
    df = df_from
    if file_from != None:
        if not os.path.isfile(file_from):
            raise FileNotFoundError(file_from)
        #if not file_from.endswith('.csv'):
         #   raise ValueError('Csv file needs to be given')
        df = pd.read_csv(file_from, header=None)
    
    y = [categories[val] for val in df[0].values]
    df_new  = pd.DataFrame(y)
    if file_to != None:
        df_new.to_csv(file_to)
    else:
        return df_new


if __name__=='__main__':
    file_from = sys.args[1]
    file_to = sys.args[2]
    if file_from!=None and file_to!=None:
        process_categories(file_from,file_to)  
    elif file_from != None:
        process_categories(file_from,file_from)
    else:
        raise FileNotFoundError('No file given')
