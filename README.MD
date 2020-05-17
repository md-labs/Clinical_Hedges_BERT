# Clinical Hedges

This repository contains the source files and data for the results produced in the paper "Cascade Neural Ensemble for Identifying
Scientifically Sound Articles" (yet to be published)

All data files / models spoken about in this repo are either present in this Dropbox link: https://www.dropbox.com/sh/5wy538ex67v7mex/AABgMCL7C9EghF-QFqZfXkcxa?dl=0
or can be reproduced using the code given 

The files given originally by the Clinical Hedges Data creators is available in this link: https://www.dropbox.com/sh/b7lpmwakig13m6s/AAA1tA1EAHl3dkWyh5sFmZbNa?dl=0

The Clinical_Hedges_Architecture.pptx contains the results and methodology used in this repo

### Source File Description
There are three folders under src:
1. Data_Extraction_And_PreProcessing: Contains code to scrape Pubmed for articles, Extract data from the excel files using Python and
to prepare the data as input to any type of model mentioned in the paper.
2. Model_Evaluation: Contains code to evaluate the different models (ITL and Cascading Type)
3. Neural_Network_Modelling: Contains Transformer code modified to be fit the various architectures described

**Data_Extraction_And_PreProcessing:**\
`Create_CrossValidation_Data_All_Models.py`- \
Input:\
MTL_Task_Labels_Marshall.csv OR MTL_Task_Labels_Del_Fiol.csv, Final_Retrieved_Clinical_Hedges.csv\
Output:\
CV_Data_ITL, CV_Data_Ensemble, CV_Data_ITL_Ensemble, CV_Data_ITL_Cascade

`Extract_CH_Articles_From_Excel.py`- \
Input:\
\
Output:\

`PubMed_Article_Extract_Scraping.py`- \
Input:\
\
Output:\

`Run_PubMed_Article_Extraction.sh`- \

**Model_Evaluation:**\
`Calculate_Cascade_CV_Score.py`- \
Input:\
\
Output:\

`Calculate_CV_Score_ITL.py`- \
Input:\
\
Output:\

`modified_classification_report.py`- \
Input:\
\
Output:\

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


### PreTrained Model Files: https://www.dropbox.com/sh/n5ts2f9yj64v7qu/AAAx_VZSI7o7OaEIVNL3tQSHa?dl=0
Has PyTorch Model Checkpoints for pretrained models such as UT_BERT, BioBERT, SciBERT which were used 
for comparison in this repo
