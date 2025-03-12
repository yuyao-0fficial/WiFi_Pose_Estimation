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

#### **Data Collection**
The CSI data in the dataset is obtained by [Linux 802.11n CSI Tool](https://github.com/spanev/linux-80211n-csitool), and the human pose data is collected by [PoseNet](https://github.com/mks0601/3DMPPE_POSENET_RELEASE) combined with stereo vision.

#### **Single_person_unit_train.py**
There are several key variables in `Single_person_unit_train.py` from line 39 to line 50: 
<table class="MsoNormalTable" border="0" cellspacing="0" cellpadding="0" style="border-collapse:collapse;mso-yfti-tbllook:1184;mso-padding-alt:0cm 0cm 0cm 0cm">
 <tbody><tr style="mso-yfti-irow:0;mso-yfti-firstrow:yes;height:12.45pt">
  <td width="123" valign="top" style="width:92.2pt;border:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:12.45pt">
  <p class="MsoNormal" align="center" style="margin-bottom:0cm;text-align:center;
  line-height:normal"><span lang="EN-US">Variable Name</span></p>
  </td>
  <td width="170" valign="top" style="width:127.55pt;border:solid windowtext 1.0pt;
  border-left:none;padding:0cm 5.4pt 0cm 5.4pt;height:12.45pt">
  <p class="MsoNormal" align="center" style="margin-bottom:0cm;text-align:center;
  line-height:normal"><span lang="EN-US">Meaning</span></p>
  </td>
  <td width="75" valign="top" style="width:56.3pt;border:solid windowtext 1.0pt;
  border-left:none;padding:0cm 5.4pt 0cm 5.4pt;height:12.45pt">
  <p class="MsoNormal" align="center" style="margin-bottom:0cm;text-align:center;
  line-height:normal"><span lang="EN-US">Value</span></p>
  </td>
  <td width="312" valign="top" style="width:233.85pt;border:solid windowtext 1.0pt;
  border-left:none;padding:0cm 5.4pt 0cm 5.4pt;height:12.45pt">
  <p class="MsoNormal" align="center" style="margin-bottom:0cm;text-align:center;
  line-height:normal"><span lang="EN-US">Effect</span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:1;height:38.5pt">
  <td width="123" rowspan="3" style="width:92.2pt;border:solid windowtext 1.0pt;
  border-top:none;padding:0cm 5.4pt 0cm 5.4pt;height:38.5pt">
  <p class="MsoNormal" align="center" style="margin-bottom:0cm;text-align:center;
  line-height:normal"><span lang="EN-US">`stg`</span></p>
  </td>
  <td width="170" rowspan="3" style="width:127.55pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:38.5pt">
  <p class="MsoNormal" align="center" style="margin-bottom:0cm;text-align:center;
  line-height:normal"><span lang="EN-US">Training Stage</span></p>
  </td>
  <td width="75" style="width:56.3pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:38.5pt">
  <p class="MsoNormal" align="center" style="margin-bottom:0cm;text-align:center;
  line-height:normal"><span lang="EN-US">0~3</span></p>
  </td>
  <td width="312" style="width:233.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:38.5pt">
  <p class="MsoNormal" style="margin-bottom:0cm;text-align:justify;text-justify:
  inter-ideograph;line-height:normal"><span lang="EN-US">Train the Autoencoder,
  each stage using the output from the previous stage as input.</span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:2;height:12.45pt">
  <td width="75" style="width:56.3pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:12.45pt">
  <p class="MsoNormal" align="center" style="margin-bottom:0cm;text-align:center;
  line-height:normal"><span lang="EN-US">4</span></p>
  </td>
  <td width="312" style="width:233.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:12.45pt">
  <p class="MsoNormal" style="margin-bottom:0cm;text-align:justify;text-justify:
  inter-ideograph;line-height:normal"><span lang="EN-US">Train the entire HPE
  model.</span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:3;height:12.45pt">
  <td width="75" style="width:56.3pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:12.45pt">
  <p class="MsoNormal" align="center" style="margin-bottom:0cm;text-align:center;
  line-height:normal"><span lang="EN-US">else</span></p>
  </td>
  <td width="312" style="width:233.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:12.45pt">
  <p class="MsoNormal" style="margin-bottom:0cm;text-align:justify;text-justify:
  inter-ideograph;line-height:normal"><span lang="EN-US">Train the localization
  module in the HPE model separately.</span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:4;height:12.45pt">
  <td width="123" rowspan="2" style="width:92.2pt;border:solid windowtext 1.0pt;
  border-top:none;padding:0cm 5.4pt 0cm 5.4pt;height:12.45pt">
  <p class="MsoNormal" align="center" style="margin-bottom:0cm;text-align:center;
  line-height:normal"><span lang="EN-US">`<span class="SpellE">continu</span>`</span></p>
  </td>
  <td width="170" rowspan="2" style="width:127.55pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:12.45pt">
  <p class="MsoNormal" align="center" style="margin-bottom:0cm;text-align:center;
  line-height:normal"><span lang="EN-US">Whether to continue the previous
  training</span></p>
  </td>
  <td width="75" style="width:56.3pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:12.45pt">
  <p class="MsoNormal" align="center" style="margin-bottom:0cm;text-align:center;
  line-height:normal"><span lang="EN-US">0</span></p>
  </td>
  <td width="312" style="width:233.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:12.45pt">
  <p class="MsoNormal" style="margin-bottom:0cm;text-align:justify;text-justify:
  inter-ideograph;line-height:normal"><span lang="EN-US">No, start training with
  random weights.</span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:5;height:12.45pt">
  <td width="75" style="width:56.3pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:12.45pt">
  <p class="MsoNormal" align="center" style="margin-bottom:0cm;text-align:center;
  line-height:normal"><span lang="EN-US">1</span></p>
  </td>
  <td width="312" style="width:233.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:12.45pt">
  <p class="MsoNormal" style="margin-bottom:0cm;text-align:justify;text-justify:
  inter-ideograph;line-height:normal"><span lang="EN-US">Yes, load existing
  weights to start training.</span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:6;height:12.45pt">
  <td width="123" style="width:92.2pt;border:solid windowtext 1.0pt;border-top:
  none;padding:0cm 5.4pt 0cm 5.4pt;height:12.45pt">
  <p class="MsoNormal" align="center" style="margin-bottom:0cm;text-align:center;
  line-height:normal"><span lang="EN-US">L_R</span></p>
  </td>
  <td width="170" style="width:127.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:12.45pt">
  <p class="MsoNormal" align="center" style="margin-bottom:0cm;text-align:center;
  line-height:normal"><span lang="EN-US">Learning Rate</span></p>
  </td>
  <td width="75" style="width:56.3pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:12.45pt">
  <p class="MsoNormal" align="center" style="margin-bottom:0cm;text-align:center;
  line-height:normal"><span lang="EN-US">0~1</span></p>
  </td>
  <td width="312" style="width:233.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:12.45pt">
  <p class="MsoNormal" style="margin-bottom:0cm;text-align:justify;text-justify:
  inter-ideograph;line-height:normal"><span lang="EN-US">Control the step size
  during gradient descent.</span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:7;height:12.45pt">
  <td width="123" rowspan="2" style="width:92.2pt;border:solid windowtext 1.0pt;
  border-top:none;padding:0cm 5.4pt 0cm 5.4pt;height:12.45pt">
  <p class="MsoNormal" align="center" style="margin-bottom:0cm;text-align:center;
  line-height:normal"><span class="SpellE"><span lang="EN-US">flag_train</span></span></p>
  </td>
  <td width="170" rowspan="2" style="width:127.55pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:12.45pt">
  <p class="MsoNormal" align="center" style="margin-bottom:0cm;text-align:center;
  line-height:normal"><span lang="EN-US">Training or not</span></p>
  </td>
  <td width="75" style="width:56.3pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:12.45pt">
  <p class="MsoNormal" align="center" style="margin-bottom:0cm;text-align:center;
  line-height:normal"><span lang="EN-US">0</span></p>
  </td>
  <td width="312" style="width:233.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:12.45pt">
  <p class="MsoNormal" style="margin-bottom:0cm;text-align:justify;text-justify:
  inter-ideograph;line-height:normal"><span lang="EN-US">No, start testing.</span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:8;height:12.45pt">
  <td width="75" style="width:56.3pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:12.45pt">
  <p class="MsoNormal" align="center" style="margin-bottom:0cm;text-align:center;
  line-height:normal"><span lang="EN-US">1</span></p>
  </td>
  <td width="312" style="width:233.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:12.45pt">
  <p class="MsoNormal" style="margin-bottom:0cm;text-align:justify;text-justify:
  inter-ideograph;line-height:normal"><span lang="EN-US">Yes, start training.</span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:9;height:12.45pt">
  <td width="123" style="width:92.2pt;border:solid windowtext 1.0pt;border-top:
  none;padding:0cm 5.4pt 0cm 5.4pt;height:12.45pt">
  <p class="MsoNormal" align="center" style="margin-bottom:0cm;text-align:center;
  line-height:normal"><span class="SpellE"><span lang="EN-US">losses_weight</span></span></p>
  </td>
  <td width="170" style="width:127.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:12.45pt">
  <p class="MsoNormal" align="center" style="margin-bottom:0cm;text-align:center;
  line-height:normal"><span lang="EN-US">One of the weights in the loss function</span></p>
  </td>
  <td width="75" style="width:56.3pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:12.45pt">
  <p class="MsoNormal" align="center" style="margin-bottom:0cm;text-align:center;
  line-height:normal"><span lang="EN-US">0~1</span></p>
  </td>
  <td width="312" style="width:233.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:12.45pt">
  <p class="MsoNormal" style="margin-bottom:0cm;text-align:justify;text-justify:
  inter-ideograph;line-height:normal"><span lang="EN-US">The weight in the loss function
  to regulate the contributions of depth and heat map. The greater the weight,
  the greater the contribution of heat map to the loss function.</span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:10;height:12.45pt">
  <td width="123" style="width:92.2pt;border:solid windowtext 1.0pt;border-top:
  none;padding:0cm 5.4pt 0cm 5.4pt;height:12.45pt">
  <p class="MsoNormal" align="center" style="margin-bottom:0cm;text-align:center;
  line-height:normal"><span class="SpellE"><span lang="EN-US">loss_posit_weight</span></span></p>
  </td>
  <td width="170" style="width:127.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:12.45pt">
  <p class="MsoNormal" align="center" style="margin-bottom:0cm;text-align:center;
  line-height:normal"><span lang="EN-US">One of the weights in the loss function</span></p>
  </td>
  <td width="75" style="width:56.3pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:12.45pt">
  <p class="MsoNormal" align="center" style="margin-bottom:0cm;text-align:center;
  line-height:normal"><span lang="EN-US">0~1</span></p>
  </td>
  <td width="312" style="width:233.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:12.45pt">
  <p class="MsoNormal" style="margin-bottom:0cm;text-align:justify;text-justify:
  inter-ideograph;line-height:normal"><span lang="EN-US">The weight in the loss
  function to regulate the contributions of localization and pose estimation.
  The greater the weight, the greater the contribution of localization to the
  loss function.</span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:11;height:12.45pt">
  <td width="123" style="width:92.2pt;border:solid windowtext 1.0pt;border-top:
  none;padding:0cm 5.4pt 0cm 5.4pt;height:12.45pt">
  <p class="MsoNormal" align="center" style="margin-bottom:0cm;text-align:center;
  line-height:normal"><span class="SpellE"><span lang="EN-US">max_weight</span></span></p>
  </td>
  <td width="170" style="width:127.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:12.45pt">
  <p class="MsoNormal" align="center" style="margin-bottom:0cm;text-align:center;
  line-height:normal"><span lang="EN-US">One of the weights in the loss function</span></p>
  </td>
  <td width="75" style="width:56.3pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:12.45pt">
  <p class="MsoNormal" align="center" style="margin-bottom:0cm;text-align:center;
  line-height:normal"><span lang="EN-US">0~1</span></p>
  </td>
  <td width="312" style="width:233.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:12.45pt">
  <p class="MsoNormal" style="margin-bottom:0cm;text-align:justify;text-justify:
  inter-ideograph;line-height:normal"><span lang="EN-US">The weight in the loss
  function to regulate the contributions of per-pixel error and peak position
  error. The greater the weight, the greater the contribution of peak position
  error to the loss function.</span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:12;height:12.45pt">
  <td width="123" style="width:92.2pt;border:solid windowtext 1.0pt;border-top:
  none;padding:0cm 5.4pt 0cm 5.4pt;height:12.45pt">
  <p class="MsoNormal" align="center" style="margin-bottom:0cm;text-align:center;
  line-height:normal"><span lang="EN-US">total</span></p>
  </td>
  <td width="170" style="width:127.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:12.45pt">
  <p class="MsoNormal" align="center" style="margin-bottom:0cm;text-align:center;
  line-height:normal"><span lang="EN-US">The total number of data</span></p>
  </td>
  <td width="75" style="width:56.3pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:12.45pt">
  <p class="MsoNormal" align="center" style="margin-bottom:0cm;text-align:center;
  line-height:normal"><span lang="EN-US">1~</span>∞</p>
  </td>
  <td width="312" style="width:233.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:12.45pt">
  <p class="MsoNormal" style="margin-bottom:0cm;text-align:justify;text-justify:
  inter-ideograph;line-height:normal"><span lang="EN-US">Used to show training
  progress.</span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:13;height:12.45pt">
  <td width="123" style="width:92.2pt;border:solid windowtext 1.0pt;border-top:
  none;padding:0cm 5.4pt 0cm 5.4pt;height:12.45pt">
  <p class="MsoNormal" align="center" style="margin-bottom:0cm;text-align:center;
  line-height:normal"><span class="SpellE"><span lang="EN-US">batch_size</span></span></p>
  </td>
  <td width="170" style="width:127.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:12.45pt">
  <p class="MsoNormal" align="center" style="margin-bottom:0cm;text-align:center;
  line-height:normal"><span lang="EN-US">Batch size in training</span></p>
  </td>
  <td width="75" style="width:56.3pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:12.45pt">
  <p class="MsoNormal" align="center" style="margin-bottom:0cm;text-align:center;
  line-height:normal"><span lang="EN-US">1~</span>∞</p>
  </td>
  <td width="312" style="width:233.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:12.45pt">
  <p class="MsoNormal" style="margin-bottom:0cm;text-align:justify;text-justify:
  inter-ideograph;line-height:normal"><span lang="EN-US">Control batch size in
  training.</span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:14;height:12.45pt">
  <td width="123" style="width:92.2pt;border:solid windowtext 1.0pt;border-top:
  none;padding:0cm 5.4pt 0cm 5.4pt;height:12.45pt">
  <p class="MsoNormal" align="center" style="margin-bottom:0cm;text-align:center;
  line-height:normal"><span class="SpellE"><span lang="EN-US">batch_val</span></span></p>
  </td>
  <td width="170" style="width:127.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:12.45pt">
  <p class="MsoNormal" align="center" style="margin-bottom:0cm;text-align:center;
  line-height:normal"><span lang="EN-US">Batch size in validation</span></p>
  </td>
  <td width="75" style="width:56.3pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:12.45pt">
  <p class="MsoNormal" align="center" style="margin-bottom:0cm;text-align:center;
  line-height:normal"><span lang="EN-US">1~</span>∞</p>
  </td>
  <td width="312" style="width:233.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:12.45pt">
  <p class="MsoNormal" style="margin-bottom:0cm;text-align:justify;text-justify:
  inter-ideograph;line-height:normal"><span lang="EN-US">Control batch size in
  validation.</span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:15;height:12.45pt">
  <td width="123" style="width:92.2pt;border:solid windowtext 1.0pt;border-top:
  none;padding:0cm 5.4pt 0cm 5.4pt;height:12.45pt">
  <p class="MsoNormal" align="center" style="margin-bottom:0cm;text-align:center;
  line-height:normal"><span class="SpellE"><span lang="EN-US">batch_test</span></span></p>
  </td>
  <td width="170" style="width:127.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:12.45pt">
  <p class="MsoNormal" align="center" style="margin-bottom:0cm;text-align:center;
  line-height:normal"><span lang="EN-US">Batch size in testing</span></p>
  </td>
  <td width="75" style="width:56.3pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:12.45pt">
  <p class="MsoNormal" align="center" style="margin-bottom:0cm;text-align:center;
  line-height:normal"><span lang="EN-US">1~</span>∞</p>
  </td>
  <td width="312" style="width:233.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:12.45pt">
  <p class="MsoNormal" style="margin-bottom:0cm;text-align:justify;text-justify:
  inter-ideograph;line-height:normal"><span lang="EN-US">Control batch size in
  testing.</span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:16;mso-yfti-lastrow:yes;height:12.45pt">
  <td width="123" style="width:92.2pt;border:solid windowtext 1.0pt;border-top:
  none;padding:0cm 5.4pt 0cm 5.4pt;height:12.45pt">
  <p class="MsoNormal" align="center" style="margin-bottom:0cm;text-align:center;
  line-height:normal"><span lang="EN-US">epoch</span></p>
  </td>
  <td width="170" style="width:127.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:12.45pt">
  <p class="MsoNormal" align="center" style="margin-bottom:0cm;text-align:center;
  line-height:normal"><span lang="EN-US">The number of training epochs</span></p>
  </td>
  <td width="75" style="width:56.3pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:12.45pt">
  <p class="MsoNormal" align="center" style="margin-bottom:0cm;text-align:center;
  line-height:normal"><span lang="EN-US">1~</span>∞</p>
  </td>
  <td width="312" style="width:233.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:12.45pt">
  <p class="MsoNormal" style="margin-bottom:0cm;text-align:justify;text-justify:
  inter-ideograph;line-height:normal"><span lang="EN-US">Control the number of
  training epochs.</span></p>
  </td>
 </tr>
</tbody>
</table>

#### **transfer.py**

