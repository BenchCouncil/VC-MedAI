# Data Access

AI.vs.Doctor database comprises a set of comma-separated value (CSV) files and all related source code. Since the database contains not only the patients’ information from MIMIC databases but also the doctors’ information from 14 medical centers, users must use the database with caution and respect. The database has been uploaded to PhysioNet35 platform  and is waiting for approval. We also upload the database to the Journal’s online submission system for review. To access the database, the following steps need to be completed:
 
    • The researchers must complete the access steps required by MIMIC databases.
    • the researchers are required to sign a data use agreement, which delineates acceptable data usage and security protocols and prohibits attempts to identify individual doctors and patients.

Our project uses the data version "Original-Recorded-Version".


# VD-MedAI Simulator

## 一.Data Embedding
| No. | Code | Description |
| ------- | ------- | ------- |
|1|requirements.txt|Install environment.|
|2|download cxr jpg.|[CxrJpg_download link](https://www.physionet.org/content/mimic-cxr-jpg/2.0.0/)|
|3|download pre-training model.|[BioBert download link](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT/tree/main) | 
|4|download original records from hospitals data.|[Original records from hospitals download link](https://drive.google.com/drive/folders/1FgkrmJ4rMx4UkInYiW5-V52ivMGcUNiZ?usp=drive_link) |
|5|cd simulator/data_process/|Change directory.|
|6|python 1_log_to_csv.py|Parsing the doctor's diagnosis logs.|
|7|python csv_to_embedding.py|Data Embedding: doctor information, model information and patient information (including imaging jpg using TorchXRayVision, imaging reports using BioBERT, and temporal examinations using TSFresh).|


## 二.The Specialized In-Silico Trials for Sepsis
### 2.1 Doctor Click Sequence Model

| No. | Code | Description |
| ------- | ------- | ------- |
|1|cd simulator/data_process/|Change directory.|
|2|python 3_final_embedding.py 'sequence'|Generating preliminary specialized model input data.|
|3|cd simulator/specialized/click_sequence/|Change directory.|
|5|python train.py|Model training.|
|6|python predict_and_eval.py|Model evaluation and prediction.|
|7|python predict_nextact.py|Predicting the percentage for the next check.The predicted results will be loaded into the final model input|


### 2.2 Doctor Diagnosis and Diagnosis Time Models (Preliminary and Final)
The final model input data will require patient advanced item Ratio to be tested according to the predicted results from the 'Doctor Click Sequence Model'.

| No. | Code | Description |
| ------- | ------- | ------- |
|1|cd simulator/data_process/|Change directory.|
|2|python 3_final_embedding.py 'sepsis_preliminary'|Generating preliminary and final specialized model input data.|
|3|python 3_final_embedding.py 'sepsis_final'|Generating preliminary and final specialized model input data.|
|4|cd simulator/specialized|Change directory.|
|5|python 1_preliminary_sepsis_diag.py 'train'|Training with the specialized model for preliminary diagnosis.|
|5|python 1_preliminary_sepsis_diag.py 'test'|Testing with the specialized model for preliminary diagnosis.|
|6|python 3_preliminary_sepsis_diagtime.py 'train'|Training with the specialized model for preliminary diagnosis time.|
|6|python 3_preliminary_sepsis_diagtime.py 'test'|Testing with the specialized model for preliminary diagnosis time.|
|7|python 4_final_sepsis_diag.py 'train'|Training with the specialized model for final diagnosis.|
|7|python 4_final_sepsis_diag.py 'test'|Testing with the specialized model for final diagnosis.|
|8|python 5_final_sepsis_diagtime.py 'train'|Training with the specialized model for final diagnosis time.|
|8|python 5_final_sepsis_diagtime.py 'test'|Testing with the specialized model for final diagnosis time.|


## 三.The Generalized In-Silico Trials for Other Diseases
model = 0h or 3h

### 3.1 Doctor Diagnosis and Diagnosis Time Models (Preliminary)
| No. | Code | Description |
| ------- | ------- | ------- |
|1|cd simulator/data_process/|Change directory.|
|1|python 3_final_embedding.py 'normal_{model}_preliminary'|Generating preliminary generalized model input data.|
|1|cd simulator/generalized/{model}/|Change directory.|
|2|python 1_preliminary_{model}_diag.py 'train'|Training with the generalized model for preliminary diagnosis.|
|2|python 1_preliminary_{model}_diag.py 'test'|Testing with the generalized model for preliminary diagnosis.|
|3|python 3_preliminary_{model}_diagtime.py 'train'|Training with the generalized model for preliminary diagnosis time.|
|3|python 3_preliminary_{model}_diagtime.py 'test'|Testing with the generalized model for preliminary diagnosis time.|


### 3.2 Patient Advanced Item Ratio to be Tested
| No. | Code | Description |
| ------- | ------- | ------- |
|1|cd simulator/data_process/|Change directory.|
|1|python 3_final_embedding.py 'normal_{model}_preliminary'|Generating preliminary generalized model input data.|
|1|cd simulator/generalized/{model}/|Change directory.|
|2|python 2_preliminary_{model}_action.py 'train'|Training with the generalized model for patient advanced item ratio to be tested.|
|2|python 2_preliminary_{model}_action.py 'test'|Testing with the generalized model for patient advanced item ratio to be tested.|
|2|python 2_preliminary_{model}_action.py 'predict'|Predicting with the generalized model for patient advanced item ratio to be tested.The predicted results will be loaded into the final model input|


### 3.3 Doctor Diagnosis and Diagnosis Time Models (Final)
The final model input data will require the predicted results from model "Patient Advanced Item Ratio to be Tested ".


| No. | Code | Description |
| ------- | ------- | ------- |
|1|cd simulator/data_process/|Change directory.|
|2|python 3_final_embedding.py 'normal_{model}_final'|Generating final generalized model input data.|
|3|cd simulator/generalized/{model}/|Change directory.|
|4|python 4_final_{model}_diag.py 'train'|Training with the generalized model for final diagnosis.|
|4|python 4_final_{model}_diag.py 'test'|Testing with the generalized model for final diagnosis.|
|5|python 5_final_{model}_diagtime.py 'train'|Training with the generalized model for final diagnosis.|
|5|python 5_final_{model}_diagtime.py 'test'|Testing with the generalized model for final diagnosis.|


# VD-MedAI Generator
| No. | Code | Description |
| ------- | ------- | ------- |
|1|cd generator/|Change directory.|
|2|python random_virdoc.py |Generate user-defined number of virtual doctors.|
|3|cd simulator/specialized/click_sequence|Change directory.|
|4|python predict_nextact_randomdoc.py |Generate patient advanced item ratio to be tested when virtual doctor diagnoses.|
|5|constant.py |Choose between Specialized 0h, Generalized 0h, or Generalized 3h.|
|6|python main_diag_randomdoc.py |Output the number of samples, the diagnosis accuracy of human doctors, and the diagnosis accuracy of virtual doctors.|
|7|python main_diagtime_randomdoc.py |Output the number of samples, the diagnosis time(min) of human doctors, and the diagnosis time(min) of virtual doctors.|
|8|python main_nextact_randomdoc.py |Output the number of samples, patient advanced item ratio to be tested when human doctors diagnose, and patient advanced item ratio to be tested when virtual doctors diagnose.|
