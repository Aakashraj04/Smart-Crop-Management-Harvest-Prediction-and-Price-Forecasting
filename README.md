# Smart Crop Management: Harvest Prediction and Price Forecasting
## 📋 Project Overview 
Harvest Prediction and Price Forecasting is an innovative project that leverages machine learning to assist farmers and stakeholders with accurate yield predictions and crop price forecasting. By analyzing environmental factors and historical data, this project aims to reduce risks, optimize agricultural practices, and maximize profits.

# Harvest Prediction :
Harvest Prediction using CNN is a machine learning project aimed at forecasting days to crop maturity based on crop images. The solution uses convolutional neural networks (CNNs) to extract features from images and predict the time remaining until harvest. This project aims to support farmers and stakeholders by optimizing planting schedules and improving planning for harvest.

## 🎯 Key Features:
-> CNN-Based Prediction: Predicts the number of days remaining until a crop is ready for harvest from crop images.<br>
-> Image Preprocessing: Resizes and normalizes images for model input.<br>
-> Train/Test Split and Evaluation: Uses Mean Absolute Error (MAE) as a metric to validate predictions.<br>
-> Crop-Specific Insights: Supports predictions across multiple crop types with subfolder-based image organization.<br>

## 🛠️ Technologies Used:

-> Python<br>
-> TensorFlow / Keras (Deep Learning Framework)<br>
-> OpenCV (Image Processing)<br>
-> NumPy, Pandas (Data Manipulation)<br>
-> Scikit-learn (Model Evaluation)<br>

## 🏗️ Workflow:

### Data Collection
Organize crop images into subfolders (e.g., /crop_images/maize, /crop_images/wheat).<br>
Each folder name represents the crop label used for prediction.<br>

### Image Preprocessing
Images are resized to 128x128 pixels.<br>
Pixel values are normalized between 0 and 1.<br>

### Model Architecture
A Convolutional Neural Network (CNN) with multiple convolution and pooling layers.<br>
Final output layer predicts days to maturity for the given crop image.<br>

### Training and Evaluation
The model is trained with an 80-20 split of the data for training and testing.<br>
Mean Absolute Error (MAE) is used to measure prediction performance.<br>

### 🚀 Future Improvements
Add More Crop Types: Expand to include more crops with labeled data.<br>
Mobile App Integration: Create a mobile app for farmers to capture and predict on-the-go.<br>
Cloud Deployment: Host the model on a cloud platform for real-time predictions.<br>
Data Augmentation: Improve model performance using augmentation techniques.<br>

# Crop Price Prediction:
Price Prediction Using Linear Regression is a machine learning project designed to predict the modal price of agricultural commodities. By analyzing various market parameters such as state, district, commodity type, and price trends, the system helps farmers, vendors, and policy-makers plan market activities more effectively.

## 🎯 Key Features:
->Multi-Category Encoding: Encodes states, districts, commodities, and other categorical data for regression analysis.<br>
->Time-Series Support: Utilizes arrival_date for capturing time-based trends.<br>
->Linear Regression Prediction: Uses a simple yet effective Linear Regression model to predict modal prices.<br>
->Data Visualization: Visualizes trends and correlations with histograms, box plots, scatter plots, and heatmaps.<br>

## 🛠️ Technologies Used:
->Python<br>
->Pandas / NumPy: Data manipulation and preprocessing<br>
->Scikit-learn: Machine learning (linear regression, metrics)<br>
->Matplotlib / Seaborn: Data visualization<br>

##  🏗️ Workflow
### Data Preprocessing
-> Encoding Categorical Data: Uses LabelEncoder to convert states, districts, markets, commodities, and varieties into numerical values.<br>
-> Timestamp Conversion: Converts the arrival_date column to Unix timestamps for compatibility in regression.<br>

### Model Architecture

-> Linear Regression Model: Trains the model using a dataset split (80% training, 20% testing).<br>
Evaluates performance using R² Score and Mean Squared Error (MSE).<br>

### Data Visualization
-> Histograms: Analyze the distribution of modal prices.<br>
-> Scatter Plots: Visualize relationships between max price and modal price.<br>
-> Box Plots: Compare the range of min, max, and modal prices.<br>
-> Heatmaps: Display correlations between numerical attributes.<br>

# 🗂️ Repository Contents
The repository includes the following files and directories:

## 📂 Data Files
-> crop_production.csv: Contains essential information about crop prices, locations, dates, and other key attributes needed for training and evaluation.

## 📁 Crop Images
-> Crop Growth Prediction Images/: Directory with organized crop images for feature extraction.<br>
-> Subfolders by crop type: jute/, maize/, rice/, sugarcane/, wheat/ .<br>
-> These images are used to build predictive models based on visual data.<br>

## 🐍 Python Scripts
-> Crop_growth_prediction.py: The main script for training and predicting crop growth with the help of images as input.<br>
-> Crop_price_prediction.py: The main script for training and predicting crop prices and containing visualizations like histograms, scatter plots, and heatmaps to understand data distributions.<br>
