Car Price Prediction with Machine Learning
Objective

Predict the selling price of a used car from its features (present price, kilometres driven, age, fuel type, transmission, seller type, owners) using regression models.

Dataset

car_data.csv has 301 cars with these columns:

Column	Meaning
Car_Name	Car model (used as brand)
Year	Year of purchase
Selling_Price	Price the car sold for (lakh) - target
Present_Price	Current showroom price (lakh)
Driven_kms	Kilometres driven
Fuel_Type	Petrol / Diesel / CNG
Selling_type	Dealer / Individual
Transmission	Manual / Automatic
Owner	Number of previous owners

Note: the dataset has no horsepower or mileage columns.

Workflow
Load data
Explore data (info, missing values, duplicates)
Clean data (remove 2 duplicates and 2 outliers)
Feature engineering (Car_Age, one-hot encoding of categorical columns)
Visualization (price distribution, scatter plot, boxplots, correlation heatmap)
Train/test split (80/20)
Train models: Linear Regression, Random Forest, Gradient Boosting (target is log-transformed because prices are skewed)
Evaluate with R2, MAE, RMSE and 5-fold cross-validation
Test a brand goodwill feature (average resale ratio per brand)
Predict the price of a new car
Results
Model	R2 (test)	MAE (lakh)	CV R2 (5-fold)
Linear Regression	0.860	1.05	0.821
Random Forest	0.893	0.82	0.938
Gradient Boosting	0.912	0.81	0.944
Key Findings
Gradient Boosting performed best.
Present_Price is by far the most important feature, followed by Car_Age. Kilometres driven has a small effect.
Diesel, automatic and dealer-sold cars sell for higher prices.
The brand goodwill feature did not improve the model (R2 0.912 vs 0.913), so it was left out of the final model.
Real-World Applications
Used-car websites suggesting listing prices
Dealers setting fair prices
Buyers checking whether a car is overpriced
Limitations
Small dataset (about 300 cars)
No horsepower or mileage data
Very expensive cars (for example Land Cruiser) are rare, so predictions for them are less reliable
Tools

Python, pandas, numpy, matplotlib, seaborn, scikit-learn

How to Run
Put car_data.csv and car_price_prediction.ipynb in the same folder.
Install the libraries: pip install pandas numpy matplotlib seaborn scikit-learn
Open the notebook in Jupyter and run all cells.
