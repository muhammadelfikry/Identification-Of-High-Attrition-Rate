# Case Study: Solving Human Resources Problems

## Business Understanding

Jaya Jaya Maju, a multinational company established in 2000 with 1000 employees, experienced difficulties in managing employees which resulted in a high attrition rate of more than 10%. 

### Business Problems
For this reason, a dashboard is needed that is useful as an insight to monitor various factors that affect the high attrition rate and a Machine Learning Model to identify employees who have the potential to leave.

### Project Scope
Implementing employee segmentation analysis using Machine Learning. The machine learning algorithm used is K-Means Clustering.

In addition, the Exploratory Data Analysis (EDA) process will also be carried out to get an overview of the dataset and create a dashboard to take insight from the existing dataset. 

### Preparation

Data source: https://github.com/dicodingacademy/dicoding_dataset/blob/main/employee/employee_data.csv

Setup environment:

Install packages
```
pip intall -r requirements.txt
```
Open the prediction.py file, replace ``` <employee.csv> ``` with the path of the file to be segmented. 
```
if __name__ == "__main__":
    dataset_path = <employee.csv>
    model_path = "./kmeans_clustering_model.joblib"
...
```
Then open the terminal/CMD write the following command to perform segmentation.
```
python prediction.py
``` 
## Business Dashboard

The total employees who affect the attrition rate are 179 employees out of a total of 1,470 employees of Jaya jaya Maju company. 60% of the total employees who affect the attrition rate are male. employees who affect the attrition rate most come from the Research & Development department with a total of 107 employees. employees with a Job Satisfaction score of 3 (High) and an average salary of $2500-$4500 below the average salary of other employees most affect the attrition rate.

Metabase

Dashboard: http://localhost:3000/public/dashboard/2cdf208c-cbfb-4cf0-97d5-f18a101ac7ae

## Conclusion

From the analysis conducted on Jaya Jaya Maju's employee data, it was found that the age factor and below-average employee salary affect the attrition rate. The machine learning model developed using the K-Means Clustering algorithm can help the HR department to segment whether an employee has the potential to leave.