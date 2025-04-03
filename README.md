# **Automatic image captioning**
Automatic image captioning is a challenging task at the intersection of computer vision and natural language processing, where a model generates meaningful descriptions for images. 

This project implements an end-to-end deep learning-based image captioning system using a Convolutional Neural Network (CNN) and a Recurrent Neural Network (RNN). The CNN (ResNet-50) extracts visual features from images, which are then processed by a Long Short-Term Memory (LSTM)-based decoder to generate captions.  

![image](https://github.com/user-attachments/assets/0445a20b-3858-4519-95f9-eb8ee098b8c4)

**Dataset:**  
 The model is trained on a large-scale dataset (MS COCO)
 - captions_train2017
 - captions_val2017
 - train2017
 - val2017
   
The project is organized into several key components: 
- **Encoder CNN Class:** Extracts feature from images using a pre-trained ResNet-50 model.
- **Decoder RNN Class:** Generates captions based on the extracted features using an LSTM network. 
- **Data Preprocessing:** Prepares the dataset for training, including image transformations and loading captions.
- **Training Script:** Trains the encoder and decoder models on the dataset.
- **Evaluation Script:** Generates captions for test images using the trained models.
 # **Testing Phase:**
 ## **EXtracted Features:** 
 ![image](https://github.com/user-attachments/assets/d8607208-3c28-436e-b4f5-89d4c33e1578)
 ## **Generated Caption:**
 ![image](https://github.com/user-attachments/assets/7799b21d-a24d-4c51-930d-a641f3557e8f)


 


