import os
import numpy as np
import pandas as pd

def generate_student_data(num_students=1200, filepath=None):
    """
    Generates a synthetic dataset of student metrics with realistic relationships
    to predict GPA (0.0 - 4.0) and Risk Level (Low, Medium, High).
    """
    np.random.seed(42)
    
    # 1. Generate base features with realistic distributions
    study_hours = np.random.uniform(2.0, 30.0, num_students)  # study hours per week
    attendance = np.random.beta(5, 1, num_students) * 60 + 40  # attendance rate (40% to 100%, left-skewed)
    sleep_hours = np.random.normal(7.0, 1.2, num_students)
    sleep_hours = np.clip(sleep_hours, 4.0, 10.0)  # average sleep hours (4 to 10)
    
    # Categorical features
    parental_support = np.random.choice(['Low', 'Medium', 'High'], size=num_students, p=[0.2, 0.5, 0.3])
    stress_level = np.random.choice(['Low', 'Medium', 'High'], size=num_students, p=[0.4, 0.4, 0.2])
    
    # Binary features
    tutoring = np.random.binomial(1, 0.3, num_students)  # 30% have tutoring
    extracurriculars = np.random.binomial(1, 0.5, num_students)  # 50% participate
    internet_access = np.random.binomial(1, 0.85, num_students)  # 85% have internet
    
    # Previous grade (40 to 100)
    previous_grade = np.random.normal(70, 12, num_students)
    previous_grade = np.clip(previous_grade, 40, 100)
    
    # 2. Map categorical to numerical factors for regression target generation
    parental_map = {'Low': 0, 'Medium': 1, 'High': 2}
    stress_map = {'Low': 2, 'Medium': 1, 'High': 0}  # high stress reduces performance
    
    parental_val = np.array([parental_map[p] for p in parental_support])
    stress_val = np.array([stress_map[s] for s in stress_level])
    
    # 3. Create prediction target: GPA (0.0 to 4.0)
    # The formula introduces logical weights for factors
    base_score = (
        0.30 * (previous_grade / 100.0) +
        0.25 * (attendance / 100.0) +
        0.20 * (study_hours / 30.0) +
        0.08 * (sleep_hours / 10.0) +
        0.05 * (parental_val / 2.0) +
        0.04 * tutoring +
        0.03 * internet_access +
        0.03 * (stress_val / 2.0) +
        0.02 * extracurriculars
    )
    
    # Add random noise
    noise = np.random.normal(0, 0.05, num_students)
    target_score = base_score + noise
    
    # Scale score to GPA (0.0 to 4.0)
    # A base score of 0.3 maps to 1.5 GPA, and 0.85 maps to 4.0 GPA
    gpa = 1.0 + (target_score - 0.2) * (3.0 / 0.7)
    gpa = np.clip(gpa, 0.0, 4.0)
    gpa = np.round(gpa, 2)
    
    # 4. Create risk label based on GPA and attendance
    # High risk: GPA < 2.0 or Attendance < 65%
    # Medium risk: 2.0 <= GPA < 3.0 or 65% <= Attendance < 80%
    # Low risk: GPA >= 3.0 and Attendance >= 80%
    risk_level = []
    for g, att in zip(gpa, attendance):
        if g < 2.0 or att < 65:
            risk_level.append('High')
        elif g < 3.0 or att < 80:
            risk_level.append('Medium')
        else:
            risk_level.append('Low')
            
    # Combine into a DataFrame
    df = pd.DataFrame({
        'study_hours': np.round(study_hours, 1),
        'attendance': np.round(attendance, 1),
        'sleep_hours': np.round(sleep_hours, 1),
        'parental_support': parental_support,
        'tutoring': tutoring,
        'extracurriculars': extracurriculars,
        'internet_access': internet_access,
        'previous_grade': np.round(previous_grade, 1),
        'stress_level': stress_level,
        'gpa': gpa,
        'risk_level': risk_level
    })
    
    if filepath:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"Generated synthetic dataset with {num_students} records at: {filepath}")
        
    return df

if __name__ == '__main__':
    # Test generation in the same folder
    current_dir = os.path.dirname(os.path.abspath(__file__))
    generate_student_data(num_students=1200, filepath=os.path.join(current_dir, 'student_data.csv'))
