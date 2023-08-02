# biobart_summarization
Model Details
Model Description
This model is a BioBart-based sequence-to-sequence model that was trained on the custom dataset to summarize radiology findings into impressions. In training i use 70000 radiology reports to train the model to summarize radiology findings into impressions.

Developed by: [Engr. Hamza Iqbal Malik (UET TAXILA)]
Shared by [optional]: [More Information Needed]
Model type: [Medical Text Summarization Model]
Language(s) (NLP): [English]
Demo: [hamzamalik11/radiology_summarizer]
Uses
The model is intended to be used for generating radiology impressions. The model should not be used for any other purpose.

Direct Use
The model can be directly used for generating impressions from radiology reports. Users can input the findings of a radiology report, and the model will generate a summarized impression based on that information.

Out-of-Scope Use
The model should not be used for any purpose other than generating impressions from radiology reports. It is not suitable for tasks outside of radiology report summarization.

Recommendations
Users should be aware of the limitations and potential biases of the model when using the generated impressions for clinical decision-making. Further information is needed to provide specific recommendations.

How to Get Started with the Model
Use the code below to get started with the model. from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSeq2SeqLM from transformers import DataCollatorForSeq2Seq

model_checkpoint = "attach your trained model here"

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint) tokenizer = AutoTokenizer.from_pretrained(model_checkpoint) from transformers import SummarizationPipeline

summarizer = SummarizationPipeline(model=model, tokenizer=tokenizer)

output= summarizer("heart size normal mediastinal hilar contours remain stable small right pneumothorax remains unchanged surgical lung staples overlying left upper lobe seen linear pattern consistent prior upper lobe resection soft tissue osseous structures appear unremarkable nasogastric endotracheal tubes remain satisfactory position atelectatic changes right lower lung field remain unchanged prior study")

Training Details
Training Data
The training data was a custom dataset of 70,000 radiology reports.The data was cleaned to remove any personal or confidential information. The data was also tokenized and normalized. The training data was split into a training set and a validation set. The training set consisted of 63,000 radiology reports, and the validation set consisted of 7,000 radiology reports.

Training Procedure
The model was trained using the Hugging Face Transformers library: https://huggingface.co/transformers/. The model was trained using the AdamW optimizer with a learning rate of 5.6e-5. The model was trained for 10 epochs.

Training Hyperparameters
Training regime: 
-[evaluation_strategy="epoch"], 
-[learning_rate=5.6e-5], 
-[per_device_train_batch_size=batch_size //4], 
-[per_device_eval_batch_size=batch_size //4,] 
-[weight_decay=0.01], -[save_total_limit=3], 
-[num_train_epochs=num_train_epochs //4], 
-[predict_with_generate=True //4], 
-[logging_steps=logging_steps], 
-[push_to_hub=False]

Evaluation
Testing Data
The testing data consisted of 10,000 radiology reports.

Factors
The following factors were evaluated: [-ROUGE-1] [-ROUGE-2] [-ROUGE-L] [-ROUGELSUM]

Metrics
The following metrics were used to evaluate the model: -[ROUGE-1 score: 44.857] -[ROUGE-2 score: 29.015] -[ROUGE-L score: 42.032] -[ROUGELSUM score: 42.038]

Results
The model achieved a ROUGE-L score of 42.032 on the testing data. This indicates that the model was able to generate summaries that were very similar to human-written summaries.



Author:
-Name: Engr. Hamza Iqbal Malik
-LinkedIn: www.linkedin.com/in/hamza-iqbal-malik-42366a239
-GitHub: https://github.com/hamza4344
