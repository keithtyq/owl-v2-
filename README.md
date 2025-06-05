# Owl-V2
Model evaluation on zero shot object detection model
Model version used : model_id = "google/owlv2-base-patch16"
[ All commit changes, updates, and bug fixes were done on a separate work account, this is just a personal showing of the project. ] 

Install requirements :
```bash
pip install -r requirements.txt
```

## Dataset :
![image](https://github.com/user-attachments/assets/cc1688c1-d199-4b77-a475-36555af8a9b2)

There is a total of 25611 annotated objects across the 2881 images used.
Dataset used is part of the COCO train 2017 set where the metadata is being extracted and streamlined for specific use case.

## Streamlit
Streamlit is used as a visualisation tool to see the annotated bounding boxes on the images used. 
We can compare the grounding truth boxes with the predicted boxes to see how the model has performed.

Run :
```bash
streamlit run streamlit.py
```

