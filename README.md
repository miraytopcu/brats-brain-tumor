# Brain Tumor Segmentation (U-Net, Attention U-Net, ResU-Net)

This project was developed to perform brain tumor segmentation from multimodal MRI scans using the BraTS 2020 dataset. Three distinct deep learning architectures—**Baseline U-Net**, **Attention U-Net**, and **ResU-Net**—were comparatively analyzed as part of the study.

## 🚀 Installation and Requirements

The necessary libraries for running the project can be installed using the following command:

```bash
!pip install -q nibabel nilearn SimpleITK

📁 Dataset and Environment Setup
The project was executed on Google Colab to leverage high GPU performance during training and evaluation. The following code is included within the notebooks to establish a Google Drive connection:

from google.colab import drive
drive.mount('/content/drive')

Note: This code is specifically configured for Google Colab. If you are running it locally or on another platform, please comment out the drive.mount lines.

🏗 Project Structure
The project follows a modular design. The core logic is maintained in .py source files, while execution and visualization are performed through .ipynb notebooks.

.
├── data/   
│   └── BraTS2020_TrainingData/
│   	  └── MICCAI_BraTS2020_TrainingData/  
│   	  	   ├── BraTS20_Training_001/
│   	  	   		 ├── ..._flair.nii
│   	  	   		 ├── ..._seg.nii
│   	  	   		 ├── ..._t1.nii
│   	  	   		 ├── ..._t1ce.nii
│   	  	   		 └── ..._t2.nii
│   	  	   ├── BraTS20_Training_002/
│   	  	   ├── BraTS20_Training_003/
│   	  	   ...
│   	  	   ├── BraTS20_Training_099/ 
│   	  	   └── BraTS20_Training_100/        
├── brats-brain-tumor/            
│   ├── models/  
│   	  ├── attention_training_log.csv    
│   	  ├── resunet_training_log.csv  
│   	  ├── unet_training_log.csv     
│   	  ├── best_attention_model.keras 
│   	  ├── best_resunet_model.keras
│   	  ├── best_unet_model.keras    
│   	  ├── final_attention_model.keras   
│   	  ├── final_resunet_model.keras  
│   	  └── final_unet_model.keras  
│   ├── notebooks/ 
│   	  ├── brats_split_summary.csv
│   	  ├── EDA.ipynb
│   	  ├── Evaluation.ipynb
│   	  ├── Preprocessing.ipynb
│   	  └── Training.ipynb          
│   ├── src/  
│   	  ├── __init__.py
│   	  ├── dataset.py
│   	  ├── eda_utils.py
│   	  ├── model.py
│   	  ├── preprocess.py
│   	  └── train.py  
│   └── README.md           
├── how_to_run.txt           
└── Project_Report.pdf

Important Note: The final code cell in Evaluation.ipynb aims to demonstrate model results on unseen data without ground truth. As per the project guidelines, the submitted data is limited to 100 samples. Consequently, this specific cell may not execute successfully on your end as it relies on an original data path not included in the compressed file. If the cell fails, please refer to the Project Report for the corresponding outputs and interpretations.

🛠 Execution Instructions (How to Run)
Modular Architecture: The .py files under the src/ directory contain the primary functions. These files are not intended for direct execution; instead, they are imported into the notebooks.

Notebook Usage: To test or train the project, open the .ipynb files in the notebooks/ folder and execute the cells sequentially.

🔗 GitHub Repository
The full source code and version history are available at:
https://github.com/miraytopcu/brats-brain-tumor

Group Members:

Miray Topcu
Delfin Aksu
Meral Özgür