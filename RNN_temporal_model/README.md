# RNN(using LSTM) based AC power prediction

## Purpose and breif overview :-

**This documentation relates to analysis and modelling the temporal pattern of power use for set of ACs and saving the optimum model for each device.** 
It helps understand general data preparation, train and inference procedure for temporal data. Understanding the flow gives an idea to implement stock, weather predictions.

The analysis involves at hour and day level power consumption to see pattern of device use, comparison/correlation between them and relation with ambient(outside) temperature. Further sine and cosine cyclicity is induced in hour level data to do model training and prediction. The 

## Detailed workflow:-

### <u>Initial analysis and observation:-</u>
(a)1st mount the colab and define path followed by setting GPU runtype environment. Read the ***"AC_Data.csv" via pandas and observe its head. It 2 month data contains daily power consumption of 18 ACs of a hotel at minute level granulation as cloumns along "time" in 0th column. Since its a client data, hence could not be uploaded.***

But same data can be simulated in differnt application. For instance, ***data set of stock pricing of "n" nifty shares or temperature of "m" differnt cities daily with minute level granulation could be created.Post that steps below could be followed.***

(b)Plot the **AC(power consumption)-time profile** for each AC. Next sum each ith column(corresponding to ith AC power consumption). This helps ***find maximum and minumum power consuming ACs.***

(c)Now rescaling of given dataframe is done at hour level granulation:- **"df_h"**. For this part-wise summation of 60 consective rows(1 hr = 60 mins) is done for each AC and new column:- **'date_cum_time_duration'** is created with date and incremental hour range.

(d)Now same plot of **AC(power consumption)-time profile** for each AC(as in step (b)) is done. Post that ***mean, deviation, deviation/mean and maximum power consumption hour duration is done for each AC.***

(e)Since we'll use the ***df_h*** for power prediction model, we can save it in ***xlsx(excel)*** file in our drive via pandas. Post this either reload ***df_h*** using pandas(named as ***df_new***)/use existing ***df_h***.

(f)Now ***download the 6 hourly updated temperature data of city of Hotel(here Gurgaon) from website:- https://www.timeanddate.com/. And add it to hourly power consumption dataframe i.e. df_h/df_new***(repeat each entry 6 consecutive hour spans to cover 6hrs). Also plot temperature-time profile plot(6 hour_level) for span of 2 months and observe profile.

(g)Now convert ***df_h/df_new*** to average day level power consumption data frame namely ***df_day***. For this take partwise mean calculation of 24 succesive rows(1 day =24 hrs) is done for each AC column + temperature. Also new column **date** which extracts date segment from **'date_cum_time_duration'** of **df_h/df_new** dataframe.

(h) **AC(power consumption)-time profile** plot for each AC+temperature is done for ***df_day***. This is followed by producing heatmap of correlation matrix(via seaborn) to:-

(i)observe relations between permutations of different AC's daily power consumption.

(ii)relation between daily power consumption of all AC's(1 by 1) with daily temperature.

Thereafter consolidated observation are noted in raw text block.

### <u>Model implementation part:-</u>

#### **(a)<u>Induce cyclicity in time:-</u>**
By creating a new dataframe from **df_h/df_new** namely **df_preprocess**. Details:-

(i) For each record extract the ***'date_cum_time_duration'*** of ***df_h/df_new***. Now split the ***date(dd:mm::yyyy format)*** from the ***hour_range***. Split the hour_range(separated by "-") and take 1st element as ***lower_hour_bound*** field. Define new field ***date_time*** as join of ***date(dd:mm::yyyy format) and lower_hour_bound(joined by " ")***. Repeat this for all records and genrate ***df_preprocess***.

(ii) **<u>Create cyclic columns</u>**:-

(1)***"day_cos"***==><u>df_preprocess["day_cos"] = [np.cos(x * (2 * np.pi / 24)) for x in df_preprocess["lower_hour_bound"]]</u> and  ***"day_sin"***==><u>df_preprocess["day_sin"] = [np.sin(x * (2 * np.pi / 24)) for x in df_preprocess["lower_hour_bound"]]</u>

(2)Create ***"timestamp"*** field using ***"date_time"***==><u>df_preprocess['timestamp'] =[datetime.timestamp(datetime.strptime(x, "%Y-%m-%d %H:%M:%S")) for x in df_preprocess["date_time"]]</u>

(3)Define **year in terms of seconds**. Create ***"month_cos"***==><u>df_preprocess["month_cos"] = [np.cos((x) * (2 * np.pi / year)) for x in df_preprocess["timestamp"]]</u> and ***"month_sin"***==><u>df_preprocess["month_sin"] = [np.sin((x) * (2 * np.pi / year)) for x in df_preprocess["timestamp"]]</u>

(iii) Rest retain all AC's columns but discard ***'date_cum_time_duration'*** from **df_h/df_new**.

================================================================
#### **(b)<u>Create "process" class and define its constructors, functions for each AC and its input values:-</u>**
(i)Define consructor and define ***hyperparameters and attributes:- "lag, n_ahead, ac_name, df, min_delta, patience, epoch, batch_size, train_days,  lr, hidden_ouput_dim, feature_column"***. Create feature_dataframe, **dfnew** by selecting given **feature(['date_time', ac_name(among 18), 'day_cos', 'day_sin', 'month_cos', 'month_sin'])** above. Below processing on **dfnew**.

----------------------------------
(ii) <u>**data_split_cum_preparation fn**</u>:- 

(1)Based on ***"train_days(<60)"*** parameter. ***Validation_days = (Total_days i.e.61-train_days) post train_days.*** 

(2)Strip ***"date_time"*** column and feed rest train & validation array for ***min_max scaling normalization*** to get values in range[0,1]. ***For this minmaxScaler() is invoked and fit onto train array. This is specif   for each AC, is saved along with fit model used for normalizion mentioned above(even on test_set during inference)***.

(3)For data preparation ***"lag" and "n_ahead" parameters are the key***. <u>***Lag*** denotes length of input X(consecutive rows of **dfnew**)</u>. While ***n_ahead*** represents length of output Y for each corresponding X that starts after row index of last element of X.

(4)Now prepare ***inputs(X_train,X_val)*** whose **"ith" input range belongs to (i, i+lag) rows and [1:] columns** and ***outputs(Y_train, Y_val)*** whose **"ith" output range belongs to (i+lag, lag+n_ahead) rows and 0th column**. The iterator **"i" hovers from (0, length of train/val array- lag-n_ahead)**.

-------------------------------------
(iii) <u> **arctitecture fn**</u>:-
This is 3 layer model with:-

(1)1st 2 as *LSTM layers(with feed_forward)* <u>(input_shape =X_train.shape) and dropout =0.2*</u>

(2)Final *dense layer* of <u>n_ahead units</u>.*

-------------------------------------
(iv)<u>**trainCallBack fn**</u>:- 
***EarlyStopping(to prevent overfit and regularize) with monitoring parameter as validation_loss(val_loss).*** Here ***delta and patience*** parameters are also used. **delta** means minumum value tolerance above which loss is considered altered, else unchanged(|delta|<"input_value"). **patience** is used where the loss stagnates to call EarlyStopping(after number of iteraions = "input_value").

(v)<u>**Training fn**</u>:-
Here loss_type is selected ***(MeanSquaredError)***, optimizer defined ***(Adam)*** and metrics to populate ***(here mean_absolute_error)***.
Now ***model is fit using attributes train & validation input, outputs, epochs, batch_size trainCallBack fn(2.iv step)*** and whole model.fit fn is assigned to <u>*history for plot(using plot fn) of train and val loss*</u>.

---------------------------------------
(v) <u>**actual_predicted_power_comparison**</u>:-

(1)Firstly ***predicted list*** of consolidated flattened o/p(**yhat**) on X_val==><u>self.model.predict(self.X_val)</u> for all rows of X_val is generated. The **Y_val** values are also flattened and consolidated to ***actual list***.

(2)**Inverse transfrom** of both lists are done to have re-scaled outputs.Now plot of both ***re-scaled actual and predicted list of o/p*** are compared on y-axis vs ***validation timeline(total_days i.e.61 - train_days)*** on x-axis using plot_fn for each AC.

-----------------------------------------
(vi)<u>***Model saved in .h5 format and minmaxScalar fitted on train_array in .pkl file</u>***.

(vii) <u>**Finally iterate through each AC model training by**</u>:-

(1)defining ***hyperparameters and attributes input value***, creating corresponding ***"process"*** class.

(2) Splitting train:validation set, normalizing their arrays via ***minmaxScaler***, preparing training and validation inputs and outputs(using ***lag and n_ahead***).

(3)Initilaizing ***"Training fn"***, training the model with ***"trainCallBack"*** instructions. Observing the **"MeanSQuareError(MSE)"** of best model at end of training .Thereafter plotting their train& validation losses. Finally comparing actual and predicted validation output on best model and plotting it. ***Saving the best model and its minmaxScalar***.

(viii)<u>**Aggreagate models(AC1-18) performance**</u>:- By observing each's final MSE--->RMSE(square root of MSE) and finding range across all(AC1-18). **Comes around 0.064 and 0.11 for AC hourly power consumption in 50-250 units.**

-------------------------------------
## Note:- In case "AC_power_prediction.ipynb" fails to unload(as > 5mb):-

**nbviewer**: Copy your GitHub notebook URL and paste it into nbviewer.org. This service renders heavy notebooks flawlessly.

**Binder**: Launch your repository in a live, interactive environment using mybinder.org.

**Google Colab**: Change the URL from ://github.com... to ://google.com... to open and run it directly in the cloud.

**Switch to Raw View**: Click the "Raw" button at the top right of the file on GitHub to download or view the text source directly.



