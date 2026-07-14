-- Total Patients
SELECT COUNT(*) AS Total_Patients
FROM patients;

-- Average Age
SELECT ROUND(AVG(Age),2) AS Average_Age
FROM patients;

-- Average BMI
SELECT ROUND(AVG(BMI),2) AS Average_BMI
FROM patients;

-- Risk Distribution
SELECT
"CVD Risk Level",
COUNT(*) AS Total
FROM patients
GROUP BY "CVD Risk Level";

-- Average Cholesterol by Risk
SELECT
"CVD Risk Level",
ROUND(AVG("Total Cholesterol (mg/dL)"),2) AS Avg_Cholesterol
FROM patients
GROUP BY "CVD Risk Level";

-- Smoking Status Count
SELECT
"Smoking Status",
COUNT(*) AS Total
FROM patients
GROUP BY "Smoking Status";