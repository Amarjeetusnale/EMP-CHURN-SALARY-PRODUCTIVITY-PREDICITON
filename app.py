import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load trained models
salary_model = joblib.load("salary_prediction_model.pkl")
churn_model = joblib.load("employee_churn_model.pkl")
productivity_model = joblib.load("employee_productivity_model.pkl")

st.set_page_config(page_title="HR Analytics Dashboard", layout="wide")
st.title("📊 HR Analytics & Prediction System")
st.markdown("Use this dashboard to predict **Salary, Churn, and Productivity** of employees.")

# Sidebar Navigation
menu = ["Home", "Salary Prediction", "Churn Prediction", "Productivity Analysis", "Upload Data"]
choice = st.sidebar.radio("Navigate", menu)

# ---------------- HOME ----------------
if choice == "Home":
    st.subheader("Welcome to HR Analytics Dashboard")
    st.image("https://cdn-icons-png.flaticon.com/512/1995/1995574.png", width=150)
    st.write("""
        This tool helps HR teams to:
        - Predict Employee Salary 💰  
        - Predict Employee Churn 🚪  
        - Analyze Productivity & Performance 📈  
    """)

# ---------------- SALARY PREDICTION ----------------
elif choice == "Salary Prediction":
    st.subheader("💰 Employee Salary Prediction")

    exp = st.number_input("Experience (Years)", 0, 40, 5)
    perf = st.slider("Performance Score", 1, 5, 3)
    edu = st.selectbox("Education Level", ["High School", "Bachelor", "Master", "PhD"])
    train = st.number_input("Training Hours", 0, 500, 20)

    edu_map = {"High School": 1, "Bachelor": 2, "Master": 3, "PhD": 4}
    edu_val = edu_map[edu]

    if st.button("Predict Salary"):
        # Use DataFrame with column names
        input_df = pd.DataFrame([[exp, perf, edu_val, train]],
                                columns=["Years_At_Company", "Performance_Score", "Education_Level", "Training_Hours"])
        pred_salary = salary_model.predict(input_df)[0]
        st.success(f"Predicted Annual Salary: $ {pred_salary:,.2f}")

# ---------------- CHURN PREDICTION ----------------
elif choice == "Churn Prediction":
    st.subheader("🚪 Employee Churn Prediction")

    age = st.number_input("Age", 18, 65, 30)
    overtime = st.number_input("Overtime Hours (Yearly)", 0, 1000, 50)
    satisfaction = st.slider("Employee Satisfaction Score", 1.0, 5.0, 3.0)
    remote = st.selectbox("Remote Work Frequency", [0, 25, 50, 75, 100])
    perf = st.slider("Performance Score", 1, 5, 3)
    train = st.number_input("Training Hours", 0, 500, 20)

    if st.button("Predict Churn"):
        input_df = pd.DataFrame([[age, overtime, satisfaction, remote, perf, train]],
                                columns=["Age", "Overtime_Hours", "Employee_Satisfaction_Score",
                                         "Remote_Work_Frequency", "Performance_Score", "Training_Hours"])
        pred_churn = churn_model.predict(input_df)[0]
        if pred_churn == 1:
            st.warning("⚠️ Employee likely to Resign!")
        else:
            st.success("✅ Employee likely to Stay")

# ---------------- PRODUCTIVITY ----------------
elif choice == "Productivity Analysis":
    st.subheader("📈 Employee Productivity & Performance")

    proj = st.number_input("Projects Handled", 0, 100, 10)
    work = st.number_input("Work Hours per Week", 20, 80, 40)
    overtime = st.number_input("Overtime Hours", 0, 500, 20)
    train = st.number_input("Training Hours", 0, 500, 30)
    satisfaction = st.slider("Employee Satisfaction Score", 1.0, 5.0, 3.5)
    team = st.number_input("Team Size", 1, 50, 8)

    if st.button("Predict Productivity Score"):
        input_df = pd.DataFrame([[proj, work, overtime, train, satisfaction, team]],
                                columns=["Projects_Handled", "Work_Hours_Per_Week", "Overtime_Hours",
                                         "Training_Hours", "Employee_Satisfaction_Score", "Team_Size"])
        pred_perf = productivity_model.predict(input_df)[0]
        st.success(f"Predicted Performance Score: {pred_perf:.2f} / 5")

# ---------------- UPLOAD DATASET ----------------
elif choice == "Upload Data":
    st.subheader("📂 Upload Employee Dataset for Insights")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())

        st.write("### Salary Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df["Monthly_Salary"], kde=True, ax=ax)
        st.pyplot(fig)

        st.write("### Attrition by Department")
        fig, ax = plt.subplots()
        sns.countplot(x="Department", hue="Resigned", data=df, ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

        st.write("### Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
