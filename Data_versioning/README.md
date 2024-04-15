`load_and_save_data` contains code for creating consistent data to be used by across different individuals, so that everyone uses the exact same data for model training and testing.
Model comparison is more precise with versioned data.

## General info about the generated data

Data is created for different usecases and are divided to their own folders **2022**, **2022_sensors**, **2023** and **2023_sensors**.

We removed high Redox_avg values (>900), so that the model only focuses on finding Redox_error_flags. High values are different kind of errors and these are removed by specialist automatically by running script using a specific threshold value.

Training and testing data are split using 70/30 rate. Training data in both 2022 folders have 57,7% of target features and testig data has 42,3% of target features.

Both 2022 folders include scaled and non-scaled versions, where scaled data is stored in it's own folder named 'Scaled'. For scaling data we used **MinMaxScaler**.

2 usecases for training (2022 and 2022_sensors) was created for performance comparison.

## 2022 data
Data includes **all pits** and **all sensors**.

## 2022_sensors data
Data includes **all pits** and is **split by different sensors** (1-5).

## Usage

Use **2022** and **2022_sensors** for model training and testing, and **2023** and **2023_sensors** only for testing models performance on 2023 data.

**Note** that 2023 data doesn't have target features (Redox_error_flag/Redox_error_flag(<pit_number>), thus model performance on this data can't be measured stastically. The prediction outcomes can be only checked visually to see if the model trained on **2022** and **2022_sensors** data finds the most obvious redox_avg fluctuations.
