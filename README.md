# Student Career Prediction using Machine Learning

This project predicts potential career paths for students based on their skills using a machine learning model.  
A Random Forest classifier is trained on student skill data to predict careers such as Software Engineer, Data Scientist, Teacher, or HR.

---

## Features Used

The model uses the following student skill attributes:

- Math score
- Programming skill
- Communication skill
- Logical reasoning

---

## Target Variable

The model predicts the following possible careers:

- Software Engineer
- Data Scientist
- Teacher
- HR

---

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

---

## Machine Learning Model

Random Forest Classifier

Random Forest is an ensemble learning method that combines multiple decision trees to improve prediction accuracy and reduce overfitting.

---

## Workflow

1. Generate a synthetic dataset of students
2. Perform data preprocessing
3. Split data into training and testing sets
4. Train a Random Forest model
5. Evaluate the model performance
6. Visualize results using graphs

---

## Results

Example model performance:

Accuracy ≈ **0.97**

The project also produces visualizations such as:

- Career distribution
- Correlation heatmap
- Confusion matrix
- Feature importance plot

---

## Project Structure


## Project Structure

student-career-prediction/
│
├── generate_dataset.py        # Script to generate synthetic student data
├── student_prediction.py      # Model training and prediction
├── README.md                  # Project documentation