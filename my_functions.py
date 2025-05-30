import pandas as pd
import datetime
import numpy as np
from scipy.signal import medfilt
from sklearn import preprocessing
from sklearn.metrics import ConfusionMatrixDisplay, balanced_accuracy_score, f1_score
import matplotlib.pyplot as plt


def read_month_data(name):
    column_interest=['TimeStamp' , 'Pressure High Drive 1  - Pressure [bar]','Pressure Low Drive 1  - Pressure [bar]', 'Actual Speed Drive 1 - Revolutions [rpm]']
    data = pd.read_csv(name, encoding="latin_1",sep=';', usecols=column_interest)
    data.rename(columns = {'TimeStamp':'Timestamp' , 'Pressure High Drive 1  - Pressure [bar]':'High-pressure', 'Pressure Low Drive 1  - Pressure [bar]': 'Low-pressure', 'Actual Speed Drive 1 - Revolutions [rpm]':'Speed'}, inplace = True)
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    data.set_index('Timestamp', inplace = True)
    data.reindex()
    return data



def ndarray_labels(start_date,end_date,timestamps, downsampled_freq):
    sampled_time = pd.Timedelta(downsampled_freq).total_seconds()
    end_date = end_date + datetime.timedelta(minutes=sampled_time/60)
    datetimes = generate_empty_datatime (start_date,end_date,sampled_time)
    num_minutes = int((end_date - start_date).total_seconds() / sampled_time)
    # Initialize an ndarray with the same shape as the list of datetime objects
    ndarray = np.empty(num_minutes, dtype='object')

    # Loop through each datetime object and assign a value to the corresponding element in the ndarray
    for start_range,end_range,label in timestamps.itertuples(index=False):
        for i, dt in enumerate(datetimes):
            if start_range <= dt <= end_range:
                ndarray[i] = label
                
    #delete isolate 
    for i in range(1, len(ndarray)-1):
        if ndarray[i]==None and ndarray[i-1]!=None and ndarray[i+1]!=None:
            ndarray[i] = ndarray[i-1]

    return ndarray

#generate a dataframe separate each 1 min

def generate_empty_datatime(start_date,end_date,sampled_time):
    datetimes = []

    # Loop through each minute between the start and end dates
    current_date = start_date
    while current_date < end_date:
        datetimes.append(current_date)
        current_date += datetime.timedelta(minutes=sampled_time/60)

    return datetimes

def smooth_labels(input_labels,kernel_size):
    df = pd.DataFrame(columns=["labels_smoothed"])
    df['labels_smoothed'] = medfilt(pd.Series(input_labels), kernel_size)

    # Map the smoothed numerical values back to categorical values
    output_labels = df['labels_smoothed'].values
    return output_labels

def smooth_labels_dutycicle(input_labels,kernel_size):

    aux = input_labels.values.astype("int")

    df = pd.DataFrame(aux,columns=["labels"])

    df['labels_smoothed'] = medfilt(df['labels'], kernel_size)

    # Map the smoothed numerical values back to categorical values
    output_labels = df['labels_smoothed'].values
    return output_labels

# convert the input file from imagimob with format: Start, Lenght, Label to Start, End and Label
def df_timestamps(labels_imagimob):
    df = pd.DataFrame(columns=["start","end","label"])
    df["label"] = labels_imagimob["Label(string)"]
    df["start"] = labels_imagimob["Time(Seconds)"].astype(np.int64) * np.int64(1e9)
    df["end"] = (labels_imagimob["Time(Seconds)"]+labels_imagimob["Length(Seconds)"]).astype(np.int64) * np.int64(1e9)
    df["start"] = pd.to_datetime(df["start"])
    df["end"] = pd.to_datetime(df["end"])
    return df


def display_results(y,y_pred,title):

    disp = ConfusionMatrixDisplay.from_predictions(y, y_pred,labels=['A','B','C','D'], colorbar=False,
                                            sample_weight=None, normalize='true',cmap=plt.cm.Blues,
                                            values_format='.2%')
    disp.ax_.set_title(title)

    #metrics
    f1_macro = f1_score(y, y_pred,average='macro',labels=['A','B','C','D'])
    balanced_acc = balanced_accuracy_score(y, y_pred)

    print(f"Balanced Accuracy: {balanced_acc:.4f} - Macro F1-Score: {f1_macro:.4f}")

    return

def report_metrics(y,y_pred):
    if not(np.issubdtype(y_pred.dtype, np.str_)):
    # Confusion Matrix
        le = preprocessing.LabelEncoder()
        le.fit(y)
        y_pred = le.inverse_transform(y_pred.astype(int)) #anti-transform int to string labels

    #metrics
    f1_macro = f1_score(y, y_pred,average='macro')
    balanced_acc = balanced_accuracy_score(y, y_pred)

    print(f"Balanced Accuracy: {balanced_acc:.4f} - Macro F1-Score: {f1_macro:.4f}")
    return

def display_results_dutycycle(y,y_pred,title):
    disp = ConfusionMatrixDisplay.from_predictions(y, y_pred, colorbar=False,
                                            sample_weight=None, normalize='true',cmap=plt.cm.Blues,
                                            values_format='.2%')
    disp.ax_.set_title(title)

    #metrics
    f1_macro = f1_score(y, y_pred,average='macro')
    balanced_acc = balanced_accuracy_score(y, y_pred)

    print(f"Balanced Accuracy: {balanced_acc:.4f} - Macro F1-Score: {f1_macro:.4f}")
    return