# Customer Segmentation

## Project Overview

This project performs customer segmentation using machine learning. The customers are grouped into different segments based on their age, income, spending score, membership years, purchase frequency, and last purchase amount.

## Dataset

The dataset contains customer information such as:

- Age
- Income
- Spending Score
- Membership Years
- Purchase Frequency
- Last Purchase Amount

The `Number` column was removed during preprocessing because it is only an ID.

## Data Preprocessing

The following steps were performed:

1. Loaded the customer dataset.
2. Checked the dataset structure and missing values.
3. Checked for duplicate rows.
4. Removed the `Number` ID column.
5. Selected the relevant customer features.
6. Standardized the features using `StandardScaler`.

## Customer Segmentation

K-Means clustering was used to divide customers into groups.

The Elbow Method was used to determine a suitable number of clusters. Based on the analysis, **4 clusters** were selected.

## Model

The K-Means model was trained using the selected customer features.

The trained model is saved in:

`model/customer_segmentation_model.pkl`

## Visualizations

The project includes the following visualizations:

- Elbow Method for Optimal K
- Customer Segments based on Income and Spending Score

The graphs are available in the `images` folder.

## Output

The final customer segmentation results are saved as:

`customer_segments.csv`

This file contains the original customer information along with the assigned cluster for each customer.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Joblib

## How to Run

Install the required packages using:

`pip install -r requirements.txt`

Then run:

`main.py`