# CSCI 5525: Homework 4
## Title: A Comparison of Transformer and LSTM Time-series Data Forecasting
### Group member: Yijun Lin (lin00786@umn.edu), Min Namgung (namgu007@umn.edu)
### Date: Dec 17, 2021


This repo is to compare Transformer and LSTM on time series forecasting
The dataset is generated based on the resource: https://github.com/CVxTz/time_series_forecasting

### Experiment 1: Transformer VS. LSTM Results:

The images demonstrates that Transformer produced more accurate prediction than LSTM. 

More detailed metrics comparison can be found below.

| Tramsformer     | LSTM model
|------------- | -------------
|![1](res_readme/T_1.png)| ![2](res_readme/LSTM_1.png)
|![3](res_readme/T_2.png)| ![4](res_readme/LSTM_2.png)


--------------------------------------------------------------------------------------------

### Experiment 2: Adding Periodic Positional Encoding on Transformer:

The syntatic data are generated based on: **A * Cos(Bx + C) + D**

where, A controls the amplitude, B is controlling the periodic of the function, e.g., 7, 14, 28, C is the horizonal shift, and D is the vertical shift
    
**So, we intentionally add periodic information in the Transformer in a postional encoding way**


| Sign  | Description
|------------- | -------------
|AUX| Auxiliary Features (e.g., Hour, Day)
|Penc| Periodic as Positional Encoding  
------------- | -----------------------------------------
|0| Not using
|1| Using

The images demonstrates that Transformer with periodic positional encoding produced more accurate prediction. 

|ID    | Transformer W/WO Positional Encoding    |
|------------- |------------- |
|Without|![5](res_readme/t_p_5.png)|
|With|![6](res_readme/t_p_6.png)|


Model| Using Auxiliary Features | Using Periodic as Positional Encoding | MAE | SMAPE
|------------- |------------- |------------- |------------- |------------- |
|LSTM|1|0|0.29625|47.51880
|Transformer|1|0|0.23089|37.93381
|Transformer|1|1|**0.19829**|**34.05033**




### How to run the code:

    Please refer to bash folder about how to train/test model


