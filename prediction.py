import joblib
import pandas as pd
import pathlib
import tarfile
from time import gmtime, strftime
import boto3

fecha=strftime("%d-%m-%Y", gmtime())

if __name__ == "__main__":
    

    # Sagemaker specific arguments. Defaults are set in the environment variables.

    
    model_path = f"/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")
    model = joblib.load("model.joblib")

    test_path = "/opt/ml/processing/output_p/data_to_predict.csv"
    df = pd.read_csv(test_path, index_col=0,sep=';')
    print(df.head(5))
    
    
    #Calcula las predicciones
    predictions = model.predict(df)
    pacientes = pd.DataFrame(df)
    results_prob = model.predict_proba(df)
    results_prob=pd.DataFrame(results_prob)
    
    #Exportar modelo
    output_dir = "/opt/ml/processing/output/"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    file_name='prediction-'+fecha+'.csv'
    evaluation_path = output_dir + file_name
    pacientes.to_csv(evaluation_path)
    s3 = boto3.client('s3')
    key="inference_model/output/"+file_name
    s3.upload_file(
    Filename=evaluation_path,
    Bucket="samay_sounds_container",
    Key=key,
    )
