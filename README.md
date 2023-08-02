# Biobart_for_summarization_of_Radiology_Reports

This model is a BioBart-based sequence-to-sequence model that was trained on the custom dataset to summarize radiology findings into impressions. In training i use 70000 radiology reports to train the model to summarize radiology findings into impressions.
![rad1](https://github.com/hamza4344/biobart_summarization/assets/62426973/2d7827da-03f1-43ea-9814-cfa275da0542)

# Uses
The model is intended to be used for generating radiology impressions. The model should not be used for any other purpose.The model can be directly used for generating impressions from radiology reports. Users can input the findings of a radiology report, and the model will generate a summarized impression based on that information.The model should not be used for any purpose other than generating impressions from radiology reports. It is not suitable for tasks outside of radiology report summarization.


# How to Get Started with the Model
# Use the code below to get started with the model. 
![code](https://github.com/hamza4344/biobart_summarization/assets/62426973/caec3ae8-b0c9-47e5-bdca-4f56bd99e2ab)


# Training Details
# Training Data
The training data was a custom dataset of 70,000 radiology reports.The data was cleaned to remove any personal or confidential information. The data was also tokenized and normalized. The training data was split into a training set and a validation set. The training set consisted of 63,000 radiology reports, and the validation set consisted of 7,000 radiology reports.

# Training Procedure
The model was trained using the Hugging Face Transformers library: https://huggingface.co/transformers/. The model was trained using the AdamW optimizer with a learning rate of 5.6e-5. The model was trained for 10 epochs.

# Training Hyperparameters
![hyper](https://github.com/hamza4344/biobart_summarization/assets/62426973/c2c23b1a-c37b-419e-a372-f3577bd3771b)


#  EVALUATION
# Testing Data
The testing data consisted of 10,000 radiology reports.

# Factors
The following factors were evaluated: 
[ROUGE-1] [ROUGE-2] [ROUGE-L] [ROUGELSUM]

# Metrics
The following metrics were used to evaluate the model: 
 [ROUGE-1 score: 44.857] 
 [ROUGE-2 score: 29.015] 
 [ROUGE-L score: 42.032] 
 [ROUGELSUM score: 42.038]

# Results
The model achieved a ROUGE-L score of 42.032 on the testing data. This indicates that the model was able to generate summaries that were very similar to human-written summaries.

# Author Details
### Developed by: [Engr. Hamza Iqbal Malik (UET TAXILA)]
### Demo: [hamzamalik11/radiology_summarizer]
### LinkedIn: www.linkedin.com/in/hamza-iqbal-malik-42366a239
### HuggingFace: https://huggingface.co/hamzamalik11
