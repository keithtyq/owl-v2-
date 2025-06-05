## Owl-V2
Model evaluation on zero shot object detection model
Model version used : model_id = "google/owlv2-base-patch16"
[ All commit changes, updates, and bug fixes were done on a separate work account, this is just a personal showing of the project. ] 


Dataset :
Data columns (total 10 columns):
 #   Column        Non-Null Count  Dtype  
---  ------        --------------  -----  
 0   bbox          25611 non-null  object 
 1   category_id   25611 non-null  int64  
 2   image_id      25611 non-null  int64  
 3   id            25611 non-null  int64  
 4   segmentation  25611 non-null  object 
 5   area          25611 non-null  float64
 6   label         25611 non-null  object 
 7   set           25611 non-null  object 
 8   filename      25611 non-null  object 
 9   split         25611 non-null  object 
dtypes: float64(1), int64(3), object(6)
There is a total of 25611 annotated objects across the 2881 images used.
Dataset used is part of the COCO train 2017 set where the metadata is being extracted and streamlined for our use case.

# Streamlit
Streamlit is used as a visualisation tool to see the annotated bounding boxes on the images used. 
We can compare the grounding truth boxes with the predicted boxes to see how the model has performed.

Run :
streamlit run streamlit.py

