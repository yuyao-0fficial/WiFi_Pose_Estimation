## **AI-Based Real-Time WiFi Human Pose Estimation by a Single Communication Chain**
This is the key source code for paper **AI-Based Real-Time WiFi Human Pose Estimation by a Single Communication Chain**


### **Quick Start**
Create and activate a new virtual environment using anaconda：
```
conda create -n #your_virtual_environment_name python=3.8 anaconda
conda activate #your_virtual_environment_name
```

Install dependencies according to `requirements.txt`:
```
conda install --yes --file requirements.txt
```

Clone the repository to local directory and enter it:
```
git clone https://github.com/yuyao-0fficial/WiFi_Pose_Estimation.git
cd WiFi_Pose_Estimation
```

Run `Single_person_unit_train.py` to invoke HPE network and estimate human pose through CSI:
```
python Single_person_unit_train.py
```

Use `transfer.py` to evaluate the estimated poses:
```
python Single_person_unit_train.py
```


### **Detailed Description**
The structure of the repository is as follows:
```
WiFi_Pose_Estimation
├── network_files
│   └── Single_Person_Estimation_unit.py                        ## HPE Network
├── README.md
├── requirements.txt
├── save_weights                                                ## Trained HPE Network Weights
│   ├── AutoEncoder_stg3_2024_12_26_18_54_37.pth
│   └── Single_Person_Estimator_4_2024_9_23_9_7_15.pth
├── single_person_annotation_5                                  ## Dataset
│   └── single person
│       ├── csi
│       │   └── test_conti.mat
│       ├── jpeg
│       │   └── test
│       │       ├── output_20231201122807_000000108.jpg
│       │       │       .
│       │       │       .
│       │       │       .
│       │       └── output_20231203143833_000000252.jpg
│       └── pose
│           └── test_conti.mat
├── Single_person_Dataset.py                                    ## Preprocessing
├── Single Person Output                                        ## The storage directory of the HPE model output
├── Single_person_unit_train.py                                 ## The code to train and test the HPE model
├── transfer.py                                                 ## The code to evaluate the estimated poses
└── utils
    └── Single_Person_Estimator_Loss_3.py                       ## Loss function
```
