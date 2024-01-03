import os
import pandas as pd
import numpy as np
import argparse
import boto3
import pathlib
from time import gmtime, strftime
import io

BASE_DIR = "ml/processing"

#Function to transform the dataset for the model
def extract_nested_zip(zip_file_path, extraction_path):
    """
    Extract a zip file, including any nested zip files.
    Delete the zip file after extraction.
    """
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        # Extract all the contents into the directory
        zip_ref.extractall(extraction_path)

        # Iterate through each file in the extracted files
        for file in zip_ref.namelist():
            # Check if the file is a zip file
            if file.endswith('.zip'):
                # Construct full path to the nested zip file
                nested_zip_path = os.path.join(extraction_path, file)

                # Recursively extract the nested zip file
                extract_nested_zip(nested_zip_path, extraction_path)

                # Optionally, remove the nested zip file after extraction
                os.remove(nested_zip_path)
    
    
#Function to extract the patients for apply the model    
def extract_df(dataset_path=f"{BASE_DIR}/inputdata",obj):
    body = obj['Body']
    with open(os.path.join(dataset_path, 'my-local-file.zip'), 'wb') as f:
        f.write(body.read())
    for file in os.listdir(dataset_path):
        if file.endswith('.zip'):
            zip_file_path = os.path.join(dataset_path, file)
            extract_nested_zip(zip_file_path, dataset_path)
    # Define the path to the dataset directory
    data_dir = pathlib.Path(dataset_path)
    
    # List the files in the directory
    commands = np.array(os.listdir(data_dir))
    
    # Filter and remove 'README.md' from the list of files
    commands = commands[commands != 'README.md']
    
    # Split each element in the list by commas, and then by underscores
    a = [line.split(',') for line in commands]
    b = [x[0].split('_') for x in a]
    
    # Extract the label only if the element has at least two parts
    label = [c[1] for c in b if len(c) > 1]
    
    # Convert labels to lowercase
    label = [x.lower() for x in label]
    
    # Iterate through the labels and perform label consolidation
    for i in range(336):
        if label[i] == 'asthma and lung fibrosis':
            label[i] = 'asthma'
        elif label[i] == 'heart failure + copd' or label[i] == 'heart failure + lung fibrosis ':
            label[i] = 'heart failure'
        else:
            label[i] = label[i]
    
    def return_unique_labels(labels):
        # Removing duplicates from the list while maintaining the order
        unique_labels = []
        for label in labels:
            if label not in unique_labels:
                unique_labels.append(label)
        return unique_labels
    
    labels = return_unique_labels(label)
    
    # Initialize an empty array to store the full paths of WAV files
    wav_files = []
    
    # Iterate through the files in the directory
    for file in os.listdir(dataset_path):
        # Check if the file is a WAV file
        if file.endswith(".wav"):
            # Add the full path of the file to the array
            wav_files.append(os.path.join(dataset_path, file))
    
    # Initialize an array to store the data of each WAV file
    wav_data = []
    
    # Iterate through the WAV file paths in the wav_files array
    for filepath in wav_files:
        # Read the WAV file using scipy.io.wavfile
        sample_rate, data = scipy.io.wavfile.read(filepath)
    
        # Add the data, sample rate, file path, and data type to the wav_data array
        wav_data.append({
            "file_path": filepath,
            "sample_rate": sample_rate,
            "data": data,
            "data_type": data.dtype
        })

    
    # Ensure the number of labels matches the number of WAV files
    assert len(label) == len(wav_data)
    
    # Extracting only the 'data' from each WAV file
    data_values = [item['data'] for item in wav_data]
    
    # Creating a DataFrame with 'data' and 'label'
    sound_df = pd.DataFrame({
        'data': data_values,  # Column 'data' containing waveform data arrays
        'label': label # Column 'label' containing labels
    
    })
    
        
    return sound_df
def extract_patient(data):
    df = data[data['label'] == 'na']]
    return df



if __name__ == "__main__":
    
    print('**************************** Inicio del proceso ****************************')
    print('Fecha del proceso: {}'.format(strftime("%d-%m-%Y", gmtime())))

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, required=True)
    args = parser.parse_args()
    input_data = args.input_data
    bucket = input_data.split("/")[2]
    key = "/".join(input_data.split("/")[3:])

    print('**************************** Carga y formateo de datos ****************************')
    pathlib.Path(f"{BASE_DIR}/input").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"{BASE_DIR}/inputdata").mkdir(parents=True, exist_ok=True)
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=bucket, Key=key)
    dataset_path=f"{BASE_DIR}/inputdata"
    data = extract_df(dataset_path=f"{BASE_DIR}/inputdata",obj)
    print("Total de datos en el dataframe: ", len(data))
    
    data_to_predict=extract_patient(data)

    output_directory = os.path.join("/opt/ml/processing/output", "data_to_train.csv")
    output_directory_to_predict = os.path.join("/opt/ml/processing/output_p", "data_to_predict.csv")
    
    print("Total de datos resultantes (DataSet Entrenamiento): ", len(data_result))
    print("Total de datos resultantes (Data para predecir): ", len(data_to_predict))


    print("Guardando datos en {}".format(output_directory))
    data.to_csv(output_directory, sep=';')
    print("Guardando datos en {}".format(output_directory_to_predict))
    data_to_predict.to_csv(output_directory_to_predict, sep=';')

    print('**************************** Fin del proceso ****************************')
