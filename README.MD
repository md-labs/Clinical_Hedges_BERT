# Clinical Hedges

This repository contains the source files and data for the results produced in the paper "Cascade Neural Ensemble for Identifying
Scientifically Sound Articles" (yet to be published)

All data files / models spoken about in this repo are either present in this Dropbox link: https://www.dropbox.com/sh/5wy538ex67v7mex/AABgMCL7C9EghF-QFqZfXkcxa?dl=0
or can be reproduced using the code given. Copy the Dataset folder to the root directory of this repo (i.e. Clinical_Hedges/) 
and change path in code appropriately for code to work.

The files given originally by the Clinical Hedges Data creators is available in this link: https://www.dropbox.com/sh/b7lpmwakig13m6s/AAA1tA1EAHl3dkWyh5sFmZbNa?dl=0

The Clinical_Hedges_Architecture.pptx contains the results and methodology used in this repo

### Source File Description
There are three folders under src:
1. Data_Extraction_And_PreProcessing: Contains code to scrape Pubmed for articles, Extract data from the excel files using Python and
to prepare the data as input to any type of model mentioned in the paper.
2. Model_Evaluation: Contains code to evaluate the different models (ITL and Cascading Type)
3. Neural_Network_Modelling: Contains Transformer code modified to be fit the various architectures described

**Data_Extraction_And_PreProcessing:**\
`Extract_CH_Articles_From_Excel.py`- Extract Clinical Hedges articles from given Excel file according to conditions set by Marshall / Del Fiol\
Input:\
Clinical Hedges Data - Medline.xls\
Output:\
All_Clincial_Hedges_Articles.csv, MTL_Task_Labels_Marshall.csv

`PubMed_Article_Extract_Scraping.py`- Given the Article ID, this code retrieves the Abstract, Title, MesH terms and PT Tags from PubMed by web scraping\
Input:\
All_Clincial_Hedges_Articles.csv\
Output:\
Final_Retrieved_Clinical_Hedges_PT.csv

`Create_CrossValidation_Data_All_Models.py`- Used to create Cross Validation for any number of folds for any of the four models described in this approach. 
Additionally, it balances the data for each task and does some preliminary preprocessing on the text\
Input:\
MTL_Task_Labels_Marshall.csv OR MTL_Task_Labels_Del_Fiol.csv, Final_Retrieved_Clinical_Hedges_PT.csv\
Output:\
CV_Data_ITL, CV_Data_Ensemble, CV_Data_ITL_Ensemble, CV_Data_ITL_Cascade

`Run_PubMed_Article_Extraction.sh`- Shell script to run PubMed_Article_Extract_Scraping.py in Agave\

**Model_Evaluation:**\
`modified_classification_report.py`- Contains sklearns classification report modified to print precision, recall and f-score with more floating point precision
(increased floating point precision to 4 decimal points)\

`Calculate_Cascade_CV_Score.py`- Used to Calculate Micro Avg F-Score and (Specificity, Precision at different recall levels by varying probabilities), AUC score
and a combined list of labels cumulating all folds for Cascade and Ensemble-Boolean\
Input:\
Folder BERT_Output_CV_ITL_Cascade (Cascade Output) OR BERT_Output_CV_ITL_Ensemble_Boolean (Ensemble-Boolean Output)\
Output:\
Overall_Cross_Validation_Report.txt, Final_Labels.csv, Fold_Wise_Classification_Report.txt


`Calculate_CV_Score_ITL.py`- Used to Calculate Micro Avg F-Score and (Specificity, Precision at different recall levels by varying probabilities), AUC score
and a combined list of labels cumulating all folds for ITL and Ensemble\
Input:\
Folder BERT_Output_CV_ITL (ITL Output) OR BERT_Output_CV_Ensemble (Ensemble Output)\
Output:\
Overall_Cross_Validation_Report.txt, Final_Labels.csv, Fold_Wise_Classification_Report.txt

`specificity.py`- From sklearn, modified the roc_curve function alone

**Neural_Network_Modelling:**\
Contains 4 folders namely: Cascade, Ensemble, Ensemble-Boolean and ITL (Individual Task Learner). Each folder represents
the four model architectures mentioned in the PPT and in the paper describing this repo (yet to be published). The folders contain
all the required files needed to run the models. 

The input data is the only change between the Cascade and Ensemble-Boolean. The Ensemble folder depends on the output of Ensemble-Boolean finetuning.

For each of the folders, the data needs to be prepared using the preprocessing code in the "Data_Extraction_And_PreProcessing" directory 
particularly the "Create_CrossValidation_Data_All_Models.py"

`run_classifier_CrossValidation_ITL.py`- Runs the ITL for the combined main\
Input:\
CV_Data_ITL\
Output:\
ITL Models for each fold in the output directory and the labels predicted as `Reqd_Labels_Part_*.csv` corresponding to each fold

`run_classifier_CrossValidation_Task.py`- Runs the ITL for individual tasks (To run for Cascade/Ensemble-Boolean and before Ensemble)\
Input:\
CV_Data_ITL_Ensemble OR CV_Data_ITL_Cascade\
Output:\
Cascade/ITL Models for each fold in the output directory and the labels predicted as `Reqd_Labels_Part_*.csv` corresponding to each fold

`run_classifier_CrossValidation_Ensemble.py`- Runs the Ensemble of individual tasks (To run after Ensemble-Boolean)\
Input:\
**CV_Data_Ensemble**, **Model Directory** containing models finetuned for individual tasks saved using the `run_classifier_CrossValidation_Task.py` code\
Output:\
Ensemble Models for each fold in the output directory and the labels predicted as `Reqd_Labels_Part_*.csv` corresponding to each fold

`Run_BERT_CH_*.sh`- Runs the text classification code for various model architectures in their respective directories


### Data Files Description: https://www.dropbox.com/sh/ni93pq6ueij0kxw/AABe_86SxoRu-9mSP3RirKfga?dl=0
`All_Clinical_Hedges_Articles.csv`- Article ID's of all Clinical Hedges articles (used to scrape from PubMed)

`Final_Retrieved_Clinical_Hedges_PT.csv`- Contains all scraped PubMed Paper data including ID, Abstract, Title, PT Tags and MesH terms

`MTL_Task_Labels_Del_Fiol.csv`- The labels for article ID's for different tasks which satisfy the DelFiol subsetting condition of the entire dataset

`MTL_Task_Labels_Marshall.csv`-  The labels for article ID's for different tasks which satisfy the Marshall subsetting condition of the entire dataset

`Classification_Reports`- Folder which contains the text classification results for Del Fiol and Marshall for different model architectures,
sampling ratios, PT tag inclusion and various other experiments.\
**Note:** Some of these results do not have the probablities recorded from the model. This is because these files were results from an earlier time
before probabilities were extracted using the code. Hence, these results might have to be re-run in order to get precision at different recall levels


### PreTrained Model Files: https://www.dropbox.com/sh/n5ts2f9yj64v7qu/AAAx_VZSI7o7OaEIVNL3tQSHa?dl=0
Has PyTorch Model Checkpoints for pretrained models such as UT_BERT, BioBERT, SciBERT which were used 
for comparison in this repo
