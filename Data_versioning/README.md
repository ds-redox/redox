`load_and_save_data` contains code for creating consistent data to be used by across different individuals, so that every uses the exact same data for model training and testing.
Model comparison is more accurate with versioned data.

Data is created for different usecases and are divided to their own folders **2022**, **2022_sensors**, **2023** and **2023_sensors**.

### Usage

Use **2022** and **2022_sensors** for model training and testing. **2023** and **2023_sensors** data is used only for testing models performance on 2023 data.
The performance on 2023 data can't be measured statistically, but rather visually to see if the model trained on **2022** and **2022_sensors** data finds the most obvious redox_avg fluctuations.
2023 data has no redox_error_flags, thus statistical measures are not possible for classification.

2 usecases for training was created for performance comparison. 

### 2022 data
Data includes all pits and all sensors. It is split for training and testing. Training data includes scaled and non-scaled versions.

### 2022_sensors data
Data includes all pits and is split by different sensors (1-5). Additionally theses splits include split for training and testing. As in 2022 this also includes scaled and non-scaled data for training.
