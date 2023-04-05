import pandas as pd
import numpy as np
import os
import sys
import time

start_time = time.time()
data = pd.read_csv('Data_Entry_2017.csv')
labels={
    'No Finding':0, 
    'Infiltration':0, 
    'Atelectasis':0, 
    'Effusion':0, 
    'Nodule':0, 
    'Pneumothorax':0, 
    'Mass':0, 
    'Cardiomegaly':0, 
    'Consolidation':0, 
    'Pleural_Thickening':0,
    'Emphysema':0,
    'Edema':0,
    'Fibrosis':0,
    'Pneumonia':0,
    'Hernia':0
    }

patient_list = []
for index, row in data.iterrows():
    # print(row)
    if row['Finding Labels'].find('Cardiomegaly') != -1:
        if row['Patient ID'] not in patient_list:
            patient_list.append(row['Patient ID'])

for index, row in data.iterrows():
    if row['Finding Labels'].find('|') == -1:
        labels[row['Finding Labels']] += 1
    else:
        words = row['Finding Labels'].split('|')
        for word in words:
            if row['Patient ID'] not in patient_list:
                labels[word] += 1
                
count = 0
for i in range(1, 13):
    folder_path = 'images_{:03d}/images/'.format(i)
    file_names = os.listdir(folder_path)
    file_names = sorted(file_names)
    for file_name in file_names:
        if file_name == data['Image Index'][count]:
            data['Image Index'][count] = folder_path + file_name
        count += 1
        sys.stdout.write('\rProgress: %d%%' % (count*100/len(data)))
        sys.stdout.flush()

                

rows = [index for index, row in data.iterrows() if row['Patient ID'] in patient_list]
data_1 = data.drop(rows)
data_1 = data_1.drop(labels=['Unnamed: 11'], axis=1)
df = data_1.to_csv('data_1.csv', index=False)
end_time = time.time()
run_time = end_time - start_time
print('Run time: %.2f seconds' % run_time)