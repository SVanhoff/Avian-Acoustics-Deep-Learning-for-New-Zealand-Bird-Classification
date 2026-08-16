# **Avian Acoustics: Deep Learning for New Zealand Bird Classification**
SHEHAN VANHOFF

## Table of Contents

- [Executive Summary](#executive-summary)
- [1. Problem Statement](#1-problem-statement)
  - [1.1 The Core Problem](#11-the-core-problem)
  - [1.2 Why This Problem is Valuable](#12-why-this-problem-is-valuable)
  - [1.3 Desired State](#13-desired-state)
  - [1.4 Previous Research](#14-previous-research)
- [2. Industry Background & Domain Scope](#2-industry-background--domain-scope)
- [3. Stakeholders](#3-stakeholders)
- [4. Business Question](#4-business-question)
  - [4.1 Main Business Question](#41-main-business-question)
  - [4.2 Business Value](#42-business-value)
- [5. Data Question](#5-data-question)
  - [5.1 Main Data Question](#51-main-data-question)
  - [5.2 Data Required](#52-data-required)
- [6. Data](#6-data)
- [7. Data Science Process](#7-data-science-process)
  - [7.1 Exploratory Data Analysis (EDA)](#71-exploratory-data-analysis-eda)
  - [7.2 Data Preprocessing & Feature Engineering](#72-data-preprocessing--feature-engineering)
  - [7.3 Modelling](#73-modelling)
    - [7.3.1 Modelling Overview](#731-modelling-overview)
    - [7.3.2 Data Pipeline](#732-data-pipeline)
    - [7.3.3 Model 1: Simple Custom CNN](#733-model-1-simple-custom-cnn)
    - [7.3.4 Model 2: Deep Custom CNN](#734-model-2-deep-custom-cnn)
    - [7.3.5 Model 3: Transfer Learning with EfficientNetV2B0](#735-model-3-transfer-learning-with-efficientnetv2b0)
    - [7.3.6 Model 4: Transfer Learning with MobileNetV2](#736-model-4-transfer-learning-with-mobilenetv2)
    - [7.3.7 Model 5: Transfer Learning with ResNet50V2](#737-model-5-transfer-learning-with-resnet50v2)
    - [7.3.8 Chosen Model](#738-chosen-model)
  - [7.4 Outcomes](#74-outcomes)
  - [7.5 Implementation](#75-implementation)
- [8. Data Answer](#8-data-answer)
- [9. Business Answer](#9-business-answer)
- [10. Response to Stakeholders](#10-response-to-stakeholders)
- [11. End-to-End Solution](#11-end-to-end-solution)
- [12. References](#12-references)

# Executive Summary
This report explores whether deep learning can automate the classification of New Zealand bird species from acoustic recordings. The project addresses the limitations of traditional bird monitoring, which is labour-intensive, costly, geographically restricted, and difficult to scale across remote conservation areas.
The study used the DOC Tier 1 New Zealand Wildlife Short Sound Crops dataset from Kaggle. From 123,086 audio clips across 103 species, the project focused on the 10 most common species, producing a final dataset of 78,054 clips. Audio files were converted, standardised, cleaned with high-pass filtering and vocalisation extraction, then transformed into Mel-spectrograms for image-based classification.
Five deep learning models were evaluated, including custom CNNs and transfer learning models. The transfer learning models performed best, with ResNet50V2 achieving the highest test accuracy at 73.48% and the lowest test loss of 0.852. This made ResNet50V2 the recommended model, while MobileNetV2 remains a practical option for lightweight edge deployment.
Overall, the project shows that automated bird-call classification is technically feasible and valuable for conservation monitoring in New Zealand. A cloud-based or hybrid edge-cloud system could help researchers and conservation managers process recordings faster, reduce monitoring costs, expand geographic coverage, and support better evidence-based conservation decisions.
# 1. Problem Statement
## 1.1 The Core Problem
New Zealand’s unique bird biodiversity is under significant threat, with many species classified as vulnerable or endangered. The Department of Conservation (DOC) and other environmental agencies face a critical challenge, effectively monitoring bird population across New Zealand’s vast and often inaccessible terrain to inform conservation strategies and assess the health of ecosystems.

Current Monitoring Challenges:
•	Labour Intensive: Manual field surveys require expert ornithologists spending hundreds of hours in the field and analysing recordings.
•	Geographically Limited: Cannot cover New Zealand’s 8+ million hectares of conservation land, especially remote and inaccessible areas.
•	Expensive: Traditional survey methods could cost thousands of dollars per site, severely limiting monitoring scope
•	Not Scalable: Cannot process the growing volume of data from automated acoustic sensors being deployed nationwide. 
## 1.2 Why This Problem is valuable
Addressing this problem carries deep ecological, economic, cultural and conservation value. Ecologically, New Zealand has lost approximately 50 of its native bird species since human arrival, making the protection of remaining populations critical. Indigenous birds act as critical indicator species, their reduction indicates environmental destruction, while on the other hand, their existence helps provide important ecological services to the ecosystem, such as pollination, seed dispersal and pest control. Economically, native avian life is an irreplaceable asset, birdwatching/wildlife tourism directly contributes millions annually to the national economy and with the Department of Conservation (DOC) allocating more than $300 million each year toward biodiversity management, efficient monitoring is necessary to optimise these resources. Culturally, native birds are taonga (treasured) species with deep significance to Māori and iconic species like the kiwi tie wildlife preservation directly to New Zealand’s national identity. Ultimately, improving monitoring capabilities creates a direct conservation impact by enabling early detection of population declines, which enables timely interventions and supporting scalable, data-driven decision for evidence-based policymaking. 
## 1.3 Desired State
The desired outcome implements an automated, continuous, nationwide monitoring system that extends into remote and hard to reach areas. By significantly reducing surveying costs, this highly scalable solution will improve accuracy through increased data collection while ensuring long term consistency.
## 1.4 Previous Research 
BirdNET: A deep learning solution for avian diversity monitoring:
Developed a DNN, called BirdNet, capable of identifying 984 North American and European bird species by sound. It achieved a mean average precision of 0.791 for single-species recordings. 
# 2. Industry Background & Domain Scope
Sitting at the intersection of Conservation Technology, Environmental Science and Ecology, this field is undergoing a major shift. As eco-acoustics takes off as its own scientific discipline, manual audio analysis is quickly giving way to automated data processing. 
The process starts in the field, where autonomous recording units capture continuous environmental audio without disturbing local wildlife. The sound files are uploaded to central database, where automated pipelines convert the audio into visual spectrograms, extracting key acoustic features and running detection models. The resulting data, maps out bird presence, distribution and trends. This empowers public agencies and non-profits to execute data-driven conservation decisions, shape environmental policy and launch targeted awareness campaigns. 
Beyond ecological monitoring, the foundational technology behind acoustic event detection and signal processing has broad cross-industry use. Its application spans smart home safety, automated security systems, automotive sound diagnostics, healthcare respiratory analysis and urban noise pollution mapping.
# 3. Stakeholders 
The primary stakeholder is The Department of Conservation (DOC). DOC’s biodiversity budget is stretched thin with 8+ million hectares of conservation land to cover, they need rapid data to respond to shifting ecosystems. Their requirement for this project is a proof of concept of bird species classification using audio clips. Expecting insights into Different Neural Network models that would be suitable for scalability. Some secondary stakeholders include Conservation NOGs, Researchers, Ecologists and Local Governments for study and public campaigns.
# 4. Business question 
## 4.1 Main Business Question 
Can an automated machine learning model accurately classify the calls of New Zealand’s most common bird species from raw audio recordings, and can this solution be practically deployed to significantly improve biodiversity monitoring efficiency?
## 4.2 Business Value 
Along with the cost reduction, monitoring capacity would increase dramatically, thousands of hours of audio processed automatically, and turnaround times reduced from months to hours.
# 5. Data question 
## 5.1 Main Data Question
Can Mel-spectrogram features, derived from bioacoustics audio recordings, effectively distinguish between 10 of New Zealand’s most common bird species?
## 5.2 Data Required 
The project relies on a straightforward dataset consisting of labelled audio files and accompanying metadata. Ideally, audio is collected in .wav format to streamline processing, with each file annotated with its primary target species. Corresponding metadata includes the filename, primary label, and common species names to facilitate clear identification throughout the pipeline.
# 6. Data 
**Primary Source:**
The Department of Conservation (DOC) Tear 1 New Zealand Wildlife Short Sound Crops was used as the primary data source for this project, acquired form Kaggle. Tear 1: Broad scale monitoring for national context is a sampling programme, which started in late 2011, it involves the regular assessment of a selection of native species and pests at location 8 km apart and spaced evenly across the landscape. 
The dataset contained 290,000 short audio clips from three different sources. Only one source was used in this project (DOC_001_Tier1). The Tier 1 dataset consisted of 123,086 audio clips from 103 different bird species. Supporting data included a 001_metadat.csv file with filename and primary label. Another csv file, bird_naming_map.csv consisted of primary label, common name and scientific name.

 <div align="center">
 	<img width="550" height="345" alt="image" src="https://github.com/user-attachments/assets/03ce525b-01df-4309-9726-13bd30b7e7de" />
</div>

**Initial Dataset:**
•	Total Audio Files: 123,086
•	Total Species: 103 unique species
•	Format: .flac
**Filtered Dataset (Top 10 Species):**
•	Total Audio Files: 78,054
•	Total Species: 10 most abundant species
•	Format: converted to .wav
•	Clips per Species: Varies from 4,373 to 15,000
To balance the dataset, morepo2 (Morepork) bird was down sampled from 30,000+ sample clips to 15,000 random clips.
**Data Split:**
 <div align="center">
	 
| Split | Percentage | Clip Count |
| :--- | :---: | :---: |
| Training | 80% | 62,443 |
| Validation | 10% | 7,805 |
| Test | 10% | 7,806 |

 </div>
 
**Metadata File: 001_metadata.csv:**
<div align="center">

| Column | Description | Example |
| :--- | :--- | :--- |
| `Primary_label` | eBird species code | `morepo2` |
| `Secondary_labels` | Additional species (if any) | `(empty)` |
| `filename` | Audio filename | `morepo2/XC14343.flac` |

</div>

**Metadata File: bird_naming_map.csv;**
<div align="center">

| Column | Description | Example |
| :--- | :--- | :--- |
| `eBird` | eBird species code | `morepo2` |
| `CommonName` | Common name | `Morepork` |
| `ScientificName` | Scientific name | `Ninox novaeseelandiae` |
| `ExtraName` | Alternative name | `Southern Boobook` |

</div>

**Audio File Properties:**
<div align="center">

| Property | Value |
| :--- | :--- |
| Format | `FLAC` → `WAV` (converted) |
| Sample Rate | `32,000 Hz` |
| Channels | `Mono` (1 channel) |
| Average Duration | `~5 seconds` |

</div>

# 7. Data science process 
## 7.1 Exploratory Data Analysis (EDA)
Started off by loading and merging the metadata CSV files, dropping redundant or unnecessary columns. The dataset was very imbalanced with morepo2 bird having 33,360 audio clips while many species such as catergr1 having only 1 sample clip. 
<div align="center">
	<img width="1001" height="413" alt="image" src="https://github.com/user-attachments/assets/b176fb5c-2f2c-4a86-a6f2-072fc2cc580b" />
</div>
The above figure shows the distribution of audio clips for the top 20 species. To address class imbalance and establish a clear proof of concept, the top 10 most prevalent bird species within the dataset was chosen for this study. Another step taken to address this was to reduce the clip count of Morepork (morepo2) to 15,000 by randomly picking the audio samples from its 33,000+ sample set. 

<div align="center">
<img width="1005" height="413" alt="image" src="https://github.com/user-attachments/assets/31f50a98-936b-41d7-8f06-13ab2275413d" />
</div>
 
Addressing class imbalance is crucial when training machine learning models because raw, unmitigated imbalance often severely degrades a model’s ability to generalise across all classes.
A new column was created in the database, full_path, stating the file path of the audio clip for each sample, then a verification check was run to see if the audio sample exists. 
A function was created to inspect the audio files to extract its metadata. This included sample rate, duration, channels, format and corruption status, these were all inserted into the data frame. From the 78,054 audio clips that was audited, none was corrupted, all had a sample rate of 32,000 Hz and a channel configuration of [1] Mono. 

<div align="center">
 <img width="258" height="378" alt="image" src="https://github.com/user-attachments/assets/a825709a-51c8-48cc-aaa4-b2bbc9514bf2" />
</div>

<div align="center">
<img width="941" height="545" alt="image" src="https://github.com/user-attachments/assets/cf29965c-9c93-41b1-b205-5ed5846f79e9" />
</div>

Looking deeper into the duration range of the audio samples, we see that the maximum duration was 36 seconds with the minimum duration being 3 seconds. The average length was around 5.3 seconds with the median being 5 second. 
## 7.2 Data Preprocessing & Feature Engineering 
Audio filtering and signal processing was conducted to improve the neural network modelling. Three distinct preprocessing functions were created, first a High-Pass Filter was used to reduce background noise. 
A High-Pass Filter is an audio processing tool designed to allow frequencies above a specific threshold to pass through while reducing frequencies below that threshold. Frequencies higher than 800 Hz was passed through. 
A Vocalisation Extraction was performed using librosa.effects.split to get bursts of sound indicating a bird call. It works by calculating the frame-by-frame amplitude of an audio signal and identifying intervals that exceed a specified noise floor. 
On average the duration of a bird call is around 2 seconds, because neural networks require fixed-size inputs the audio length was standardized. This was done by trimming audio or padding shorter audio with zeros to exactly 3 second. All these steps were fitted into a pipeline for easy processing. 
Using these filtered audio, Mel-Spectrograms were extracted. Mel-Spectrogram is a visual representation of an audio signal that shows how the frequencies of the sound changes over time. It is essentially a heat map of the audio file where the x-axis represents time, the y-axis is pitch and the colour intensity is the logarithmic amplitude.

<div align="center">
<img width="939" height="584" alt="image" src="https://github.com/user-attachments/assets/46843438-e11b-42e0-b03e-af258e82ec23" />
</div>
 
The above figure shows the comparison between raw and filtered audio, in its Waveform and Mel-Spectrogram. Looking at the raw and filtered Waveform, the noise has been drastically reduced, leaving identifiable bird calls. The raw Mel-Spectrogram shows bright orange/yellow band running across the very bottom of the image. This is low-frequency background noise. Looking at the filtered Mel-Spectrogram, due to the High-Pass Filter, the bottom band is gone. Because of this, the actual bird vocalisations, the bright horizontal lines, stand out against the background. This is important because CNN models rather than learning in the “wind” patterns, will focus only on the bird calls.

 <div align="center">
<img width="940" height="236" alt="image" src="https://github.com/user-attachments/assets/56b98bd9-6be4-4aa1-a112-9c64f8ad795d" />
<img width="940" height="239" alt="image" src="https://github.com/user-attachments/assets/a9802628-ec5c-4cab-9927-a5877cfc374c" />


</div>

The above comparison shows the difference in the sound features of the two species. While the amplitude of the Blackbird (eurbla) only reaches around ±0.03, the Grey Warbler’s (gryger1) amplitude reaches ±0.15. This results in a very different Mel-Spectrograms.
The cleaned and balanced dataset was exported to a CSV file. The audio processing functions, including the filters and Mel-Spectrogram functions were saved to a helper module so it could be easily imported to the model training notebook.
## 7.3 Modelling 
### 7.3.1 Modelling Overview 
The modelling phase evaluated five different deep learning architectures for bird species classification. The models ranged from simple custom convolutional neural networks to advanced transfer learning approaches using pre-trained models. 

<div align="center">

| Model | Architecture | Key Features |
| :--- | :--- | :--- |
| **Model 1** | Simple Custom CNN | 2 Convolutional layers, Flatten, Dropout |
| **Model 2** | Deep Custom CNN | 4 Convolutional layers, Batch Normalisation, Global Average Pooling |
| **Model 3** | EfficientNetV2B0 (Transfer Learning) | Pre-trained on ImageNet (14M+ images) |
| **Model 4** | MobileNetV2 (Transfer Learning) | Lightweight, mobile-optimised architecture |
| **Model 5** | ResNet50V2 (Transfer Learning) | Deep residual architecture with skip connections |

</div>

### 7.3.2 Data Pipeline 
Before model training, the complete dataset was configured using TensorFlow’s data API for efficient loading and preprocessing. 
The audio file paths were extracted from the cleaned balanced data frame, and the species labels were encoded as integers. The dataset was split maintaining class proportions across all splits. 80% training split for model learning, 10% validation split for hyperparameter tuning and overfitting monitoring and 10% test split for the final evaluation of the model performance.
The preprocessing pipeline included audio loading, high-pass filtering, Vocalisation Extraction, audio length standardisation, mel-spectrogram extraction, log transformation of the spectrogram to compress the dynamic range and channel addition to prepare the data as a grayscale image for the convolution neural networks. 
### 7.3.3 Model 1: Simple Custom CNN 
The first model established a baseline performance benchmark. Input spectrograms were resized to 64x64 pixels to speed up training while preserving essential features. A normalisation layer standardised input values for training stability.
Two convolutional blocks were used, the first with 32 filters (3x3, ReLU) and max pooling, the second with 64 filters (3x3, ReLU) and max pooling with dropout (0.25). Features were flattened and passed through a dense layer with 128 neurons and ReLU activation, followed by drop (0.5) and the 10-neuro output layer.
Training: Adam optimiser, sparse categorical cross-entropy loss, 10 epochs
Results:
<div align="center">
	 <img width="940" height="291" alt="image" src="https://github.com/user-attachments/assets/39b25a8b-77e9-4fbd-9a7a-602a1db20332" />

</div>


The training accuracy started off with 42% and gradually increased to 61% by epoch 10 while the validation accuracy started off with 51% and ended up at 62% by epoch 10. The training loss started at 1.7 and by the end of the run, decreased down to 1.15. The validation loss started a bit lower than by epoch 10 it was around the same as the training loss. 
<div align="center">
<img width="546" height="390" alt="image" src="https://github.com/user-attachments/assets/b56b8d88-8701-4f03-86af-a3f7e67e00d4" />
</div>
The model achieved a final accuracy of 62% while its best performing class, class 4, had a recall of 86% and a precision of 94%.
<div align="center">
<img width="694" height="722" alt="image" src="https://github.com/user-attachments/assets/19d8d198-b31c-4328-a28f-3436acc9e536" />
</div>
The morepo2 (class 4) predicted 1294 out of its 1500 test sample clips. Gryger1 has the highest misclassifications on a single class, predicting 153 samples of being timtit1. The lowest true positives were riflem1, with 185 predicted correctly.

### 7.3.4 Model 2: Deep Custom CNN  
The second model significantly extended the baseline with a deeper architecture and multiple regularization techniques. Input spectrograms were kept at full resolution to preserve all acoustic details.

Four convolutional blocks with progressively increasing filters were used:

- **Block 1**: 32 filters (3x3, padding = `same`), Batch Normalisation, ReLU, MaxPooling
- **Block 2**: 64 filters (3x3, padding = `same`), Batch Normalisation, ReLU, MaxPooling
- **Block 3**: 128 filters (3x3, padding = `same`), Batch Normalisation, ReLU, MaxPooling, Dropout (0.3)
- **Block 4**: 256 filters (3x3, padding = `same`), Batch Normalisation, ReLU, MaxPooling, Dropout (0.3)

Instead of flattening, Global Average Pooling reduced each feature map to a single value, dramatically reducing parameters while preserving spatial information. The classifier head consisted of Dense (256), Batch Normalisation, ReLU, Dropout (0.5) followed by the output layer.
Training: Adam optimiser, sparse categorical cross-entropy loss, 10 epochs
Results: 

<div align="center">
	<img width="940" height="287" alt="image" src="https://github.com/user-attachments/assets/b4f3f2fc-93e2-48cd-b2a1-23579da350c0" />
</div>

The validation accuracy and the loss started of consistent, but at epoch 5 it had a dramatic change. By epoch 6 the callback function was triggered, which stops the training if the validation loss increases for two epochs. 
<div align="center">
	<img width="552" height="392" alt="image" src="https://github.com/user-attachments/assets/86439174-2860-480a-99f6-1624afe8922a" />
</div>

The accuracy was slightly lower than Model 1 at 60%. The recall of class 4 was lower than Model 1 with 83% compared to 86% but the precision of class 4 increased from 94% to 96%.
<div align="center">
	<img width="791" height="809" alt="image" src="https://github.com/user-attachments/assets/08f3d3f7-fa4c-4554-bb29-29deae5d83c4" />
</div>
The confusion matrix indicates that Model 2 made large scale misclassification across several class pairs. In eight instances, at least 100 samples from a given true class were assigned to a single incorrect predicted class.

### 7.3.5 Model 3: Transfer Learning with EfficientNetV2B0 
EfficientNetV2B0, pre-trained in ImageNet, was adapted for bird species classification by duplicating the single-channel spectrogram to three channels to match the RGB input format expected by EfficientNet. A two-phase fine-tuning approach was employed. In phase 1, the base model was frozen while the custom classification head was trained for 10 epochs. The head consisted of Global Average Pooling, Dense (256, ReLu), and Dropout (0.6). In phase 2, the base model was unfrozen and fine-tuned with a lower learning rate (1e-5) for up to 30 epochs with early stopping to prevent overfitting.
<div align="center">
	<img width="940" height="283" alt="image" src="https://github.com/user-attachments/assets/e496cd32-4553-413b-a5f0-6e8e6f83e4e1" />
</div>

The accuracy of training and validation increased rapidly increased in the first few epochs while the loss plummeted down. The validation accuracy and loss were performing better during the training till epoch 15, where the validation accuracy and loss steadied out ultimately triggering the early stopper at epoch 22. 
<div align="center">
	<img width="544" height="386" alt="image" src="https://github.com/user-attachments/assets/ee3db9a2-84be-4346-9446-22ad8fc25758" />
</div>

The model achieved 73.19% test accuracy. As expected, the best recall and precision was for class 4 with 94%. Surprisingly class 3 did better than expected with 89% recall and 93% precision. This was similar to every other model, class 3 outperforms with a lower test sample.  
<div align="center">
	<img width="632" height="653" alt="image" src="https://github.com/user-attachments/assets/87060dd4-89e3-46b3-8f53-5950255018c1" />
</div>
Looking at class 3 (lotkoe1), we see that it was able to predict 541 out of its 608 samples.

### 7.3.6 Model 4: Transfer Learning with MobileNetV2 
MobileNetV2 is a lightweight architecture optimised for mobile and edge deployment, using depthwise separable convolutions to reduce parameters while maintaining performance. The single channel spectrogram was duplicated to three channels. 
Unlike the EfficientNetV2 model, MobileNetV2 was trained using a single-phase approach with the base model unfrozen from the start. The base model was set as trainable, and the classifier head was trained simultaneously. The classification head consisted of Global Average Pooling, Dense (256, ReLU), Batch Normalisation, and Dropout (0.5). The model was complied with learning rate of 1e-4 and trained for up to 10 epochs with early stopping.
 
**Results:** 
<div align="center">
	<img width="940" height="286" alt="image" src="https://github.com/user-attachments/assets/bb3e0a42-e6da-4222-be4a-834fb30e0bae" />
</div>

The validation loss was decreasing till epoch 3, then increased and early stopping was triggered at epoch 6.
<div align="center">
	<img width="549" height="395" alt="image" src="https://github.com/user-attachments/assets/1fae1ef4-2af0-45a7-a5cc-c52dbe62c827" />
</div>

MobileNetV2 achieved an accuracy of 69.52% with 2,589,514 parameters with its best performers being class 3 and 4. Class 3 had a recall of 86% and a precision of 93%. Class 4 in the other hand and a recall of 86% and a precision of 98%. 
<div align="center">
	<img width="606" height="623" alt="image" src="https://github.com/user-attachments/assets/1e63c31c-5e92-418b-a8e5-09072f382f21" />
</div>
Looking at the confusion matrix, you see that the morepo2 (class 4) predicted 1268 out of 1500 and lotkoe1 (class 3) predicted 524 out of its 608. 

### 7.3.7 Model 5: Transfer Learning with ResNet50V2  
ResNet50V2 is a deep residual network with 50 layers and skip connections that enable training of very deep networks. The single channel spectrogram was duplicated to three channels. The base model was set as trainable from the start, allowing end-to-end fine-tuning of all layers. The classification head consisted of Global Average Pooling, Dense (256, ReLU), and a high Dropout (0.6) to prevent overfitting. 
**Results:**
<div align="center">
	<img width="940" height="286" alt="image" src="https://github.com/user-attachments/assets/354d2519-ff34-420e-a02e-70bd23d6c5eb" />
</div>
<div align="center">
	<img width="549" height="388" alt="image" src="https://github.com/user-attachments/assets/2f25e39f-ae0f-4628-bbfb-d846471e407b" />
</div>

The model achieved 73.48% test accuracy, the highest among all the models tested. Class 4 performs remarkably well, achieved 94% precision and recall, whereas class 7 struggled with only 37% recall. This is as expected with class 4 having 1500 test samples and class 7 only having 437.

<div align="center">
	<img width="755" height="788" alt="image" src="https://github.com/user-attachments/assets/1a8d9e2d-5101-41cf-b6ca-bb50a509e368" />
</div>

Looking at class 4 and 7 in the confusion matrix, class 4 predicted 1416 out of its 1500 whereas class 7 only predicted 163 out of its 437.


<div align="center">

| Metric | Model 1 | Model 2 | Model 3 | Model 4 | Model 5 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Test Accuracy** | 62.45% | 59.61% | 73.19% | 69.52% | **73.48%** |
| **Test Loss** | 1.145 | 1.271 | 0.869 | 0.974 | **0.852** |
| **Parameters** | 1.6M | 0.4M | 6.2M | 2.5M | 24.0M |
| **Training Time** | 1h 57m | 2h 08m | 10h 26m | 2h 52m | 4h 37m |

</div>

### 7.3.8 Chosen Model
## 7.4 Outcomes 
The modelling phase successfully demonstrated that deep learning can accurately classify New Zealand bird species from bioacoustics recordings. Among the five models evaluated, ResNet50V2 achieved the highest test accuracy of 73.48%, confirming that transfer learning from ImageNet provides robust features that transfer effectively to spectrogram classification. The comparison revealed that all three transfer learning models (EfficientNetV2B0, MobileNetV2, and ResNet50V2) significantly outperformed custom CNN models. Key model insights included the importance of Batch Normalisation for training stability and GlobalAveragePooling2D for preventing memory errors.
The project also confirmed the critical importance of data preprocessing and class balancing. High-pass filtering (800 Hz cutoff) was essential for removing low-frequency environmental noise that would otherwise mask bird vocalizations. The results demonstrated that an automated acoustic monitoring system is both feasible and practical for deployment in New Zealand’s conservation programs. The selected ResNet50V2 model provides a robust foundation for future expansion to additional species and real-world field deployment. 
## 7.5 Implementation 
Deploying the bird classification model into production requires careful consideration of integration with existing workflow, and ongoing maintenance. The selected ResNet50V2 model can be deployed as a cloud-based API to process audio files uploaded by researchers, with predicted species and confidence scores returned via a simple dashboard interface. For scenarios requiring real-time processing on field sensors, the lightweight MobileNetV2 model offers a practical alternative for edge deployment on devices like Raspberry pi. A hybrid approach, edge device performing initial filtering followed by cloud-base detailed classification, could provide the best of both worlds. Predictions should be exportable for integration with DOC’s internal systems, and regular model retraining with new labelled data will maintain and improve performance over time.
# 8. Data answer 
Can Mel-spectrogram features, derived from bioacoustics audio recordings, effectively distinguish between 10 of New Zealand’s most common bird species?
Yes, the ResNet50V2 model achieved 73.49% test accuracy in the 10 species classification task, demonstrating that Mel-Spectrogram features contain sufficient information to distinguish between bird species with high reliability. The evaluation was performed on a test set that the model never saw during training, representing real-world audio with environmental noise, variable recording conditions, and authentic bird vocalizations. The consistency between validation and test performance indicates the model generalised well to new, unseen data.
# 9. Business answer
Can an automated machine learning model accurately classify the calls of New Zealand’s most common bird species from raw audio recordings, and can this solution be practically deployed to significantly improve biodiversity monitoring efficiency?
Yes, with high confidence. The project successfully demonstrates that automated bird species classification is both technically feasible and practically deployable. With 73.48% test accuracy, the system meets the business requirements for common species monitoring while offering substantial cost savings, scalability, and fast response times compared to traditional manual surveys. This solution provides the Department of Conservation with a powerful tool to enhance biodiversity monitoring cross New Zealand, supporting evidence-base conservation decisions and contributing to the protection of the country's unique avian heritage. 
# 10. Response to stakeholders 
The automated bird classification system represents a significant step forward in biodiversity monitoring for New Zealand. By leveraging advanced deep learning technology, this solution empowers stakeholders to make faster, more informed conservation decisions, ultimately contributing to the protection and preservation of New Zealand’s unique avian heritage for future generations. We recommend immediate action on the pilot deployment to begin realising these benefits and to gather real-world feedback for continued improvement. 
# 11. End-to-end solution 
The complete end-to-end solution enables automated bird species classification from raw audio recordings through a streamlined pipeline. The process begins with audio collection, where acoustic sensors deployed in the fields capture continuous recordings of New Zealand’s natural soundscapes. These recordings are stored as audio files and transferred to a central repository for processing.
The second stage involves data preprocessing. Each audio clip is loaded at 32,000 Hz, passed through a high-pass filter with an 800 Hz cutoff to remove low-frequency environmental noise such as wind, rain, and river flow. A a vocalisation extraction is conducted and standardised to 3 seconds in length. The cleaned audio is then converted into Mel-Spectrogram, which serves as the visual representation of the audio signal. A final channel dimension is added to format the spectrogram as a grayscale image suitable for CNN input. This entire preprocessing pipeline is automated and handles a variability of real-world fields recordings.
The third stage is model prediction using the selected ResNet50V2 model. The pre-trained model processed the Mel-Spectrogram and outputs probability scores for each of the 10 bird species. 
The fourth stage delivers the results to end users. A user-friendly web dashboard allows researchers, conservation managers, and other stakeholders to upload audio files, view classification results, and access confidence scores. All predictions and metadata are stored in a searchable database, enabling historical analysis and trend tracking.
As new labelled data becomes available from DOC’s ongoing monitoring programs, the model can be retrained periodically to maintain and improve performance. This end-to-end solution is designed to be scalable, cost-effective, and seamlessly integrated into existing conservation workflow, ultimately providing the Department of Conservation and other stakeholders with powerful tool for biodiversity monitoring across New Zealand.
# 12. References
1.	https://www.landcareresearch.co.nz/discover-our-research/environment/sustainable-society-and-policy/garden-birds-the-science-behind-the-survey

2.	https://www.doc.govt.nz/nature/native-animals/birds/

3.	https://www.forestandbird.org.nz/

4.	https://www.sciencelearn.org.nz/resources/1157-protecting-native-birds

5.	https://www.doc.govt.nz/our-work/monitoring-and-reporting-system/

6.	https://blog.tepapa.govt.nz/2015/07/28/extinct-birds-of-new-zealand-part-1-a-diverse-menagerie-sadly-departed/

7.	https://www.landcareresearch.co.nz/assets/researchpubs/LC1731-implementing-inventory-monitoring-programme-DOC.pdf

8.	https://www.doc.govt.nz/globalassets/documents/about-doc/role/managing-conservation/assessing-the-value-of-pcl/tourism-economy-on-pcl.pdf

9.	https://www.doc.govt.nz/news/budget/budget-2025-overview/

10.	https://www.stats.govt.nz/news/our-indigenous-species-are-at-risk-of-extinction/

11.	https://www.sciencedirect.com/science/article/pii/S1574954121000273

12.	https://www.karolpiczak.com/papers/Piczak2015-ESC-ConvNet.pdf

13.	https://www.rnz.co.nz/news/environment/332000/four-out-of-five-nz-bird-species-in-trouble

