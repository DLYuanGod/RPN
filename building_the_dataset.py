import data_pre_processing
import re
from sklearn.preprocessing import LabelEncoder
import pandas as pd

encoder = LabelEncoder()


def data_cleaning(data):
    data = data.apply(lambda z: data_pre_processing.remove_punctuations(z))
    data = data.apply(lambda z: data_pre_processing.remove_html(z))
    data = data.apply(lambda z: data_pre_processing.remove_url(z))
    data = data.apply(lambda z: data_pre_processing.remove_emoji(z))
    data = data.apply(lambda z: data_pre_processing.remove_abb(z))
    data = data.apply(lambda x: ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",x).split()))
    return data

def label_encoder(data,model='encoder'):
    if model == 'encoder':
        data = encoder.fit_transform(data)
    else:
        data = encoder.inverse_transform(data)
    return data


def building_dataset(path,islabel=1):
    dataset = pd.read_csv(path, sep=',')
    data_text = dataset['text']
    if islabel:
        data_label = label_encoder(dataset['label'])
        return data_text,data_label
    return data_text