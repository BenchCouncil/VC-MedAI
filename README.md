# VC-MedAI---Establishing Rigorous and Cost-effective Clinical Trials for AI in Medicine

# 1. Data Access to AI.vs.Clinician and MIMIC databases.

VC-MedAI is constructed based on AI.vs.Clinician, which comprises a set of comma-separated value (CSV) files indicating the interactions between AI and clinicians. 

To access the AI.vs.Clinician database, the following steps need to be completed:
 
    • The researchers must complete the access steps required by MIMIC databases.
    • the researchers are required to sign a data use agreement, which delineates acceptable data usage and security protocols and prohibits attempts to identify individual clinicians and patients.
    • The researchers are required to send an access request to the contributors and provide a description of the research project.

Our project uses the data version "Original-Recorded-Version" within AI.vs.Clinician database. The patient cohort uses MIMIC databases.

1.PhysioNet Account

First, apply for an account on the PhysioNet platform. Then, proceed to take the CITI PROGRAM exam and obtain the exam report. With the exam report in hand, you can apply for database usage permission on the MIMIC official website.

After completing the aforementioned steps, you can download the relevant MIMIC datasets.

Notice: To access AI.vs.Clinician database, one more step is needed: send an access request to the contributors and provide a description of the research project.

2.AI.vs.Clinician Database

AI.vs.Clinician is a large and human-centered database that comprises information related to the behavior variations of clinicians’ diagnosis with or without the assistance of different AI models.

3.MIMIC-III 1.4 Database

MIMIC-III is a large-scale clinical intensive care database that is freely available and openly shared. The data covers the period from 2001 to 2012.

4.MIMIC-IV 2.2 Database

MIMIC-IV is an updated version of MIMIC-III, with data covering the period from 2008 to 2019.

5.MIMIC-CXR-JPG 2.0.0 Database

The MIMIC-CXR-JPG database is a collection of chest X-ray images in JPG format, corresponding to patients in the MIMIC-IV dataset.

6.MIMIC-IV-NOTE 2.2 Database

The MIMIC-IV-Note primarily consists of discharge summaries and imaging text reports for patients in the MIMIC-IV dataset.

# 2.VC-MedAI Simulator

## 2.1 Data Embedding
| No. | Code | Description |
| ------- | ------- | ------- |
|1|requirements.txt|Install environment.|
|2|project_path.py|Modify to current project path.|
|3|download cxr jpg.|[CxrJpg_download link](https://www.physionet.org/content/mimic-cxr-jpg/2.0.0/)|
|4|download pre-training model.|[BioBert download link](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT/tree/main) | 
|6|cd simulator/data_process/|Change directory.|
|7|python 1_log_to_csv.py|Parsing the clinicians's diagnosis logs.|
|8|python csv_to_embedding.py|Patient information embedding(including imaging jpg using TorchXRayVision, imaging reports using BioBERT, and temporal examinations using TSFresh).|


## 2.2 VC-MedAI Specialized Simulator for Sepsis
### 2.2.1 Clinician Click Sequence of Viewed Examination Items Model

| No. | Code | Description |
| ------- | ------- | ------- |
|1|cd simulator/data_process/|Change directory.|
|2|python 3_final_embedding.py 'sequence'|Generating preliminary specialized model input data.(Including Clinician, Model and Patient information.)|
|3|cd simulator/specialized/click_sequence/|Change directory.|
|5|python train.py|Model training.|
|6|python predict_and_eval.py|Model evaluation and prediction.|
|7|python predict_nextact.py|Predicting the percentage for the next check.The predicted results will be loaded into the final model input|


### 2.2.2 Clinician Diagnosis and Diagnosis Time Models (Preliminary and Final)
The final model input data will require patient advanced item Ratio to be tested according to the predicted results from the 'Clinician Click Sequence Model'.

| No. | Code | Description |
| ------- | ------- | ------- |
|1|cd simulator/data_process/|Change directory.|
|2|python 3_final_embedding.py 'sepsis_preliminary'|Generating preliminary and final specialized model input data.(Including Clinician, Model and Patient information.)|
|3|python 3_final_embedding.py 'sepsis_final'|Generating preliminary and final specialized model input data.(Including Clinician, Model and Patient information.)|
|4|cd simulator/specialized|Change directory.|
|5|python 1_preliminary_sepsis_diag.py 'train'|Training with the specialized model for preliminary diagnosis.|
|5|python 1_preliminary_sepsis_diag.py 'test'|Testing with the specialized model for preliminary diagnosis.|
|6|python 3_preliminary_sepsis_diagtime.py 'train'|Training with the specialized model for preliminary diagnosis time.|
|6|python 3_preliminary_sepsis_diagtime.py 'test'|Testing with the specialized model for preliminary diagnosis time.|
|7|python 4_final_sepsis_diag.py 'train'|Training with the specialized model for final diagnosis.|
|7|python 4_final_sepsis_diag.py 'test'|Testing with the specialized model for final diagnosis.|
|8|python 5_final_sepsis_diagtime.py 'train'|Training with the specialized model for final diagnosis time.|
|8|python 5_final_sepsis_diagtime.py 'test'|Testing with the specialized model for final diagnosis time.|


## 2.3 VC-MedAI Generalized Simulator for Medicine
Generalized simulator provides general simulation and has no patient sector.

model = 0h or 3h

### 2.3.1 Clinician Diagnosis and Diagnosis Time Models (Preliminary)
| No. | Code | Description |
| ------- | ------- | ------- |
|1|cd simulator/data_process/|Change directory.|
|1|python 3_final_embedding.py 'normal_{model}_preliminary'|Generating preliminary generalized model input data.(Including Clinician, Model and Patient information.)|
|1|cd simulator/generalized/{model}/|Change directory.|
|2|python 1_preliminary_{model}_diag.py 'train'|Training with the generalized model for preliminary diagnosis.|
|2|python 1_preliminary_{model}_diag.py 'test'|Testing with the generalized model for preliminary diagnosis.|
|3|python 3_preliminary_{model}_diagtime.py 'train'|Training with the generalized model for preliminary diagnosis time.|
|3|python 3_preliminary_{model}_diagtime.py 'test'|Testing with the generalized model for preliminary diagnosis time.|


### 2.3.2 Patient Advanced Item Ratio to be Tested Model
| No. | Code | Description |
| ------- | ------- | ------- |
|1|cd simulator/data_process/|Change directory.|
|1|python 3_final_embedding.py 'normal_{model}_preliminary'|Generating preliminary generalized model input data.(Including Clinician, Model and Patient information.)|
|1|cd simulator/generalized/{model}/|Change directory.|
|2|python 2_preliminary_{model}_action.py 'train'|Training with the generalized model for patient advanced item ratio to be tested.|
|2|python 2_preliminary_{model}_action.py 'test'|Testing with the generalized model for patient advanced item ratio to be tested.|
|2|python 2_preliminary_{model}_action.py 'predict'|Predicting with the generalized model for patient advanced item ratio to be tested.The predicted results will be loaded into the final model input.|


### 2.3.3 Clinician Diagnosis and Diagnosis Time Models (Final)
The final model input data will require the predicted results from model "Patient Advanced Item Ratio to be Tested ".


| No. | Code | Description |
| ------- | ------- | ------- |
|1|cd simulator/data_process/|Change directory.|
|2|python 3_final_embedding.py 'normal_{model}_final'|Generating final generalized model input data.(Including Clinician, Model and Patient information.)|
|3|cd simulator/generalized/{model}/|Change directory.|
|4|python 4_final_{model}_diag.py 'train'|Training with the generalized model for final diagnosis.|
|4|python 4_final_{model}_diag.py 'test'|Testing with the generalized model for final diagnosis.|
|5|python 5_final_{model}_diagtime.py 'train'|Training with the generalized model for final diagnosis time.|
|5|python 5_final_{model}_diagtime.py 'test'|Testing with the generalized model for final diagnosis time.|


# 3.VC-MedAI Clinician Generator
| No. | Code | Description |
| ------- | ------- | ------- |
|1|cd generator/|Change directory.|
|2|python randomdoc_num.py |Generate the number of user-defined virtual clinicians with new features through stratified sampling.|
|3|python randomdoc_logcsv.py | Parsing the virtual clinician's diagnosis logs.|
|4|python randomdoc_csv_to_embedding.py |Patient information embedding(including imaging jpg using TorchXRayVision, imaging reports using BioBERT, and temporal examinations using TSFresh). |
|5|python randomdoc_sepsis_nextact_predict.py |Predicting the advanced item ratio for patients using virtual clinician. The predicted results will be loaded into the final model input.|
|6|python randomdoc_first_final_embedding.py |Generating model input data. (Including Virtual Clinician, Model and Patient information.)|
| |cd generator/ramdomdoc_analyze/ |Switch Path.|
|7|python randomdoc_constant.py |Choose VC-MedAI Simulator model between Specialized 0h, Generalized 0h, or Generalized 3h. |
|8|python main_diag_randomdoc.py |Output the number of samples, the diagnosis accuracy of virtual clinician. |
|9|python main_diag_truedoc.py |Output the number of samples, the diagnosis accuracy of human clinician.|
|10|python main_diagtime_randomdoc.py|Output the number of samples, the diagnosis time of virtual clinician. |
|11|python main_diagtime_truedoc.py |Output the number of samples, the diagnosis time of human clinician. |
|12|python main_normal_nextact_randomdoc.py |Output examination item percentage to be viewed for final diagnosis of virtual clinician. |
|13|python main_normal_nextact_truedoc.py |Output examination item percentage to be viewed for final diagnosis of human clinician. |
