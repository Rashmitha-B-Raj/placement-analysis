import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import os
import bcrypt
from io import StringIO

# -------------------------------
# App Title / Banner
# -------------------------------
st.set_page_config(page_title="College Placement & Career Portal", layout="wide")
st.markdown("""
# 🎓 College Placement & Career Portal
Welcome to your AI-assisted placement analysis and career guidance system.
""")

# -------------------------------
# CSS Styling
# -------------------------------
st.markdown("""
<style>
.main {
    background-color: #f9f9f9;
    font-family: 'Trebuchet MS', sans-serif;
}
h1, h2, h3 {
    color: #2e8b57;
    font-weight: bold;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 8px;
    padding: 10px 20px;
}
.stTextInput>div>input {
    border: 2px solid #4CAF50;
    border-radius: 6px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Load trained models and encoders
# -------------------------------
# These files must exist in the same directory as app.py
MODEL_FILES = {
    "clf": "placement_model.pkl",
    "reg": "salary_model.pkl",
    "le_branch": "branch_encoder.pkl",
    "le_gender": "gender_encoder.pkl",
    "expected_order": "expected_order.pkl"
}

missing_files = [v for v in MODEL_FILES.values() if not os.path.exists(v)]
if missing_files:
    st.warning("Some model/encoder files are missing. The app will still run but predictions will be disabled until the files are available.")
    for f in missing_files:
        st.info(f"Missing: {f}")

# Safe loads with fallback
def safe_load(path):
    try:
        return joblib.load(path)
    except Exception:
        return None

clf = safe_load(MODEL_FILES["clf"])
reg = safe_load(MODEL_FILES["reg"])
le_branch = safe_load(MODEL_FILES["le_branch"])
le_gender = safe_load(MODEL_FILES["le_gender"])
expected_order = safe_load(MODEL_FILES["expected_order"])

# If expected_order is None, provide a reasonable default order used across the app
if expected_order is None:
    expected_order = [
        "CGPA", "Backlogs", "Internships", "Certifications", "Projects",
        "Programming_Skills", "Age", "Aptitude_Test_Score", "Communication_Score",
        "Branch", "Gender"
    ]

# -------------------------------
# User Account System (with bcrypt)
# -------------------------------
USER_FILE = "users.csv"

if not os.path.exists(USER_FILE):
    pd.DataFrame(columns=["username","password"]).to_csv(USER_FILE, index=False)

def register_user(username, password):
    if not username or not password:
        return False, "Username and password cannot be empty."
    users = pd.read_csv(USER_FILE)
    if username in users["username"].values:
        return False, "Username already exists."
    hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    new_row = pd.DataFrame({"username":[username], "password":[hashed_pw.decode('utf-8')]})
    users = pd.concat([users, new_row], ignore_index=True)
    users.to_csv(USER_FILE, index=False)
    return True, "Account created."

def validate_user(username, password):
    users = pd.read_csv(USER_FILE)
    user_row = users[users["username"] == username]
    if user_row.empty:
        return False
    stored_hash = user_row.iloc[0]["password"].encode('utf-8')
    try:
        return bcrypt.checkpw(password.encode('utf-8'), stored_hash)
    except Exception:
        return False

# -------------------------------
# Preprocessing Function (safe encoding)
# -------------------------------
def safe_label_encode(series, le):
    """
    Encode a pandas Series using a LabelEncoder-like object.
    If unseen labels are present, map them to a new code (len(classes)).
    If encoder is None, return zeros.
    """
    if le is None:
        return pd.Series([0]*len(series), index=series.index)
    try:
        # If le has transform, use it
        encoded = le.transform(series)
        return pd.Series(encoded, index=series.index)
    except Exception:
        # fallback: map known classes, unknown -> new code
        try:
            classes = list(le.classes_)
            mapping = {c: i for i, c in enumerate(classes)}
            default_code = len(classes)
            return series.map(lambda x: mapping.get(x, default_code)).astype(int)
        except Exception:
            return pd.Series([0]*len(series), index=series.index)

def preprocess_csv(df, le_branch, le_gender, expected_order):
    # Rename if needed
    df = df.copy()
    df.rename(columns={
        "Skills": "Programming_Skills",
        "Aptitude": "Aptitude_Test_Score",
        "Communication": "Communication_Score"
    }, inplace=True)

    # Drop irrelevant columns safely
    df.drop(["Student_ID","Placed","Salary"], axis=1, inplace=True, errors="ignore")

    # Ensure numeric columns exist and fillna
    numeric_cols = ["CGPA", "Backlogs", "Internships", "Certifications", "Projects",
                    "Programming_Skills", "Age", "Aptitude_Test_Score", "Communication_Score"]
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Encode categorical safely
    if "Branch" in df.columns:
        df['Branch'] = safe_label_encode(df['Branch'].astype(str), le_branch)
    else:
        df['Branch'] = 0

    if "Gender" in df.columns:
        df['Gender'] = safe_label_encode(df['Gender'].astype(str), le_gender)
    else:
        df['Gender'] = 0

    # Add missing columns
    for col in expected_order:
        if col not in df.columns:
            df[col] = 0

    # Reorder to expected_order
    df = df[expected_order]
    return df

# -------------------------------
# Session State Initialization
# -------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None

# -------------------------------
# Sidebar: Login/Register + Logout
# -------------------------------
st.sidebar.header("🔑 Account Access")
choice = st.sidebar.radio("Choose Action", ["Login", "Register"])

if choice == "Register":
    new_user = st.sidebar.text_input("New Username")
    new_pass = st.sidebar.text_input("New Password", type="password")
    if st.sidebar.button("Create Account"):
        ok, msg = register_user(new_user, new_pass)
        if ok:
            st.success("✅ Account created! Please login.")
        else:
            st.error(f"⚠️ {msg}")

elif choice == "Login":
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        if validate_user(username, password):
            st.success(f"Welcome, {username}!")
            st.session_state.logged_in = True
            st.session_state.username = username
        else:
            st.error("Invalid credentials")

# Logout button
if st.session_state.logged_in:
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.success("You have been logged out.")

# -------------------------------
# Knowledge Base for AI Tutor (expanded)
# -------------------------------
# Each topic contains: answer (str), comparison (DataFrame), viz (DataFrame)
def build_qa_bank():
    qa = {}

    qa["data analytics"] = {
        "answer": "📊 Data Analytics focuses on extracting insights from raw data using statistics, visualization, and basic ML techniques. It emphasizes transforming data into actionable business insights.",
        "comparison": pd.DataFrame({
            "Aspect": ["Data Analytics", "Data Science"],
            "Focus": ["Analyzing existing data", "Building predictive models"],
            "Tools": ["Excel, Tableau, SQL", "Python, R, ML libraries"],
            "Career Path": ["Business Analyst", "Data Scientist"]
        }),
        "viz": pd.DataFrame({
            "Skill": ["SQL", "Python", "Tableau", "Excel"],
            "Demand (%)": [85, 90, 70, 65]
        })
    }

    qa["cloud computing"] = {
        "answer": "☁️ Cloud Computing delivers computing services (servers, storage, databases, networking, software) over the internet. It enables scalable, on-demand resources.",
        "comparison": pd.DataFrame({
            "Aspect": ["Cloud", "On-Premise"],
            "Cost": ["Pay-as-you-go", "High upfront"],
            "Scalability": ["Elastic", "Limited"],
            "Best For": ["Dynamic workloads", "Regulated/stable workloads"]
        }),
        "viz": pd.DataFrame({
            "Provider": ["AWS", "Azure", "Google Cloud"],
            "Market Share (%)": [32, 22, 10]
        })
    }

    qa["cybersecurity"] = {
        "answer": "🔐 Cybersecurity protects systems and networks from digital attacks. It includes threat detection, incident response, and secure architecture.",
        "comparison": pd.DataFrame({
            "Aspect": ["Cybersecurity", "Data Privacy"],
            "Focus": ["Prevent breaches", "Protect personal data"],
            "Tools": ["Firewalls, IDS", "Encryption, Policies"],
            "Career Path": ["Security Analyst", "Privacy Officer"]
        }),
        "viz": pd.DataFrame({
            "Threat": ["Phishing", "Malware", "Ransomware"],
            "Frequency (%)": [45, 30, 25]
        })
    }

    qa["ai ml"] = {
        "answer": "🤖 AI/ML builds systems that learn from data. ML focuses on algorithms and models; AI is broader and includes reasoning and planning.",
        "comparison": pd.DataFrame({
            "Aspect": ["Supervised", "Unsupervised", "Reinforcement"],
            "Goal": ["Predict outcomes", "Find patterns", "Learn via rewards"],
            "Examples": ["Regression, Classification", "Clustering", "Game agents"]
        }),
        "viz": pd.DataFrame({
            "Library": ["TensorFlow", "PyTorch", "Scikit-learn"],
            "Usage (%)": [45, 35, 20]
        })
    }

    qa["resume"] = {
        "answer": "📝 A strong resume highlights measurable achievements: projects, internships, tools used, and impact. Keep it concise and tailored to the role.",
        "comparison": pd.DataFrame({
            "Resume Type": ["Project-heavy", "Certification-heavy"],
            "Strength": ["Shows initiative & practical skills", "Shows formal learning & specialization"],
            "Best For": ["Startups, R&D", "Corporate, compliance roles"]
        }),
        "viz": pd.DataFrame({
            "Section": ["Education", "Projects", "Internships", "Skills"],
            "Importance (%)": [30, 30, 20, 20]
        })
    }

    qa["interview"] = {
        "answer": "💡 Use the STAR method (Situation, Task, Action, Result) for behavioral answers. For technical rounds, practice coding, system design, and problem-solving.",
        "comparison": pd.DataFrame({
            "Interview Type": ["Technical", "HR", "Managerial"],
            "Focus": ["Coding & systems", "Fit & behavior", "Strategy & leadership"],
            "Preparation": ["Practice problems", "Prepare stories", "Understand business"]
        }),
        "viz": pd.DataFrame({
            "Stage": ["Technical", "HR", "Managerial"],
            "Weightage (%)": [50, 30, 20]
        })
    }

    qa["aptitude"] = {
        "answer": "📊 Aptitude tests measure logical reasoning, quantitative ability, and verbal skills. Regular timed practice improves speed and accuracy.",
        "comparison": pd.DataFrame({
            "Section": ["Math", "Logical Reasoning", "Verbal"],
            "Difficulty": ["High", "Medium", "Medium"],
            "Preparation": ["Quant practice", "Puzzles & reasoning", "Reading & vocab"]
        }),
        "viz": pd.DataFrame({
            "Section": ["Math", "Logical Reasoning", "Verbal"],
            "Weightage (%)": [40, 35, 25]
        })
    }

    qa["soft skills"] = {
        "answer": "🤝 Soft skills like communication, teamwork, and leadership are critical. They often determine long-term career growth and team fit.",
        "comparison": pd.DataFrame({
            "Skill": ["Communication", "Teamwork", "Leadership"],
            "Why It Matters": ["Conveys ideas clearly", "Delivers results with others", "Guides teams & decisions"],
            "How to Improve": ["Practice presentations", "Group projects", "Lead small initiatives"]
        }),
        "viz": pd.DataFrame({
            "Skill": ["Communication", "Teamwork", "Leadership"],
            "Importance (%)": [40, 35, 25]
        })
    }

    qa["emerging tech"] = {
        "answer": "🚀 Emerging tech includes Blockchain, IoT, AR/VR, and Edge Computing. These areas create niche roles and cross-disciplinary opportunities.",
        "comparison": pd.DataFrame({
            "Tech": ["Blockchain", "IoT", "AR/VR"],
            "Use Cases": ["Decentralized apps", "Connected devices", "Immersive experiences"],
            "Skills Needed": ["Cryptography, Solidity", "Embedded systems, MQTT", "3D modeling, Unity"]
        }),
        "viz": pd.DataFrame({
            "Tech": ["Blockchain", "IoT", "AR/VR"],
            "Adoption (%)": [20, 35, 15]
        })
    }

    qa["career paths"] = {
        "answer": "📈 Career paths vary by interest: Data Analyst, Data Scientist, ML Engineer, Cloud Engineer, Security Analyst, DevOps Engineer, etc.",
        "comparison": pd.DataFrame({
            "Role": ["Data Analyst", "Data Scientist", "Security Analyst"],
            "Focus": ["Dashboards & reporting", "Modeling & research", "Protecting systems"],
            "Avg Salary (LPA)": [6, 12, 8]
        }),
        "viz": pd.DataFrame({
            "Role": ["Data Analyst", "Data Scientist", "Security Analyst"],
            "Avg Salary (LPA)": [6, 12, 8]
        })
    }

    return qa

qa_bank = build_qa_bank()

# -------------------------------
# Main App Content (only if logged in)
# -------------------------------
if st.session_state.logged_in:
    # ✅ Define all four tabs here
    tab1, tab2, tab3 = st.tabs([
    "🧑 Student Entry", 
    "👩‍🏫 Teacher Upload", 
    "📄 Resume Builder & ATS Analyzer"
])

    # -------------------------------
    # Tab 1: Student Entry
    # -------------------------------
    with tab1:
        st.header("Student Placement Prediction")

        col1, col2 = st.columns(2)
        with col1:
            cgpa = st.number_input("CGPA", 0.0, 10.0, 7.5, step=0.1)
            backlogs = st.number_input("Backlogs", 0, 10, 0)
            internships = st.number_input("Internships", 0, 5, 1)
            certifications = st.number_input("Certifications", 0, 10, 2)
            projects = st.number_input("Projects", 0, 10, 2)
        with col2:
            skills = st.slider("Programming Skills (1-10)", 1, 10, 7)
            age = st.number_input("Age", 18, 30, 22)
            aptitude = st.slider("Aptitude Test Score (0-100)", 0, 100, 75)
            communication = st.slider("Communication Score (1-10)", 1, 10, 7)
            branch = st.selectbox("Branch", ["CSE", "ECE", "EEE", "ME"])
            gender = st.selectbox("Gender", ["Male", "Female"])

        # Build input_data aligned with expected_order
        # Encode branch/gender safely
        branch_encoded = safe_label_encode(pd.Series([branch]), le_branch).iloc[0]
        gender_encoded = safe_label_encode(pd.Series([gender]), le_gender).iloc[0]

        input_row = {
            "CGPA": cgpa,
            "Backlogs": backlogs,
            "Internships": internships,
            "Certifications": certifications,
            "Projects": projects,
            "Programming_Skills": skills,
            "Age": age,
            "Aptitude_Test_Score": aptitude,
            "Communication_Score": communication,
            "Branch": branch_encoded,
            "Gender": gender_encoded
        }

        # Ensure all expected columns present
        input_data = pd.DataFrame([[input_row.get(col, 0) for col in expected_order]], columns=expected_order)

        if st.button("Predict Placement"):
            if clf is None or reg is None:
                st.error("Prediction models are not loaded. Place model files in the app directory.")
            else:
                try:
                    placement = clf.predict(input_data)[0]
                    salary = reg.predict(input_data)[0]
                    # Show textual result
                    if placement == 1:
                        st.success(f"✅ Student is likely to be placed.\n💰 Estimated Salary: {salary:.2f} LPA")
                    else:
                        st.error("❌ Student is not likely to be placed.")

                    # Placement probability visualization (if available)
                    if hasattr(clf, "predict_proba"):
                        prob = clf.predict_proba(input_data)[0]
                        fig, ax = plt.subplots()
                        ax.pie(prob, labels=["Not Placed","Placed"], autopct='%1.1f%%', colors=['#ff6b6b','#4CAF50'])
                        ax.set_title("Placement Probability")
                        st.pyplot(fig)
                    else:
                        st.info("Model does not provide probability scores.")

                    # Skill visualization
                    skills_data = pd.DataFrame({
                        "Category": ["Programming", "Aptitude", "Communication"],
                        "Score": [skills, round(aptitude/10, 2), communication]
                    })
                    st.subheader("Skill Snapshot")
                    st.bar_chart(skills_data.set_index("Category"))

                    # Quick comparison vs hypothetical cohort averages (example)
                    cohort_avg = pd.DataFrame({
                        "Metric": ["CGPA", "Programming", "Aptitude", "Communication"],
                        "You": [cgpa, skills, aptitude/10, communication],
                        "Cohort Avg": [7.0, 6.5, 6.8, 6.5]
                    })
                    st.subheader("You vs Cohort Averages")
                    st.table(cohort_avg)

                except Exception as e:
                    st.error(f"⚠️ Error during prediction: {e}")

    # -------------------------------
    # Tab 2: Teacher Upload
    # -------------------------------
    with tab2:
        st.header("Teacher Bulk Upload & Analysis")

        uploaded_file = st.file_uploader("Upload student dataset (CSV)", type="csv")
        if uploaded_file is not None:
            try:
                # Read CSV
                df = pd.read_csv(uploaded_file)
                st.write("Preview of uploaded data:")
                st.dataframe(df.head())

                # Preprocess safely
                df_processed = preprocess_csv(df.copy(), le_branch, le_gender, expected_order)

                # Predict using only expected features
                if clf is None or reg is None:
                    st.error("Prediction models are not loaded. Place model files in the app directory.")
                else:
                    df['Predicted_Placed'] = clf.predict(df_processed)
                    df['Predicted_Salary'] = reg.predict(df_processed)

                    st.write("📊 Bulk Prediction Results (first 50 rows):")
                    display_cols = [c for c in ["CGPA","Branch","Predicted_Placed","Predicted_Salary"] if c in df.columns]
                    st.dataframe(df[display_cols].head(50))

                    # Placement rate
                    placement_rate = df['Predicted_Placed'].value_counts().sort_index()
                    # Ensure both classes present
                    if 0 not in placement_rate.index:
                        placement_rate.loc[0] = 0
                    if 1 not in placement_rate.index:
                        placement_rate.loc[1] = 0
                    placement_rate = placement_rate.sort_index()

                    fig, ax = plt.subplots()
                    placement_rate.plot(kind='bar', color=['#ff6b6b','#4CAF50'], ax=ax)
                    ax.set_title("Placement Rate (Predicted)")
                    ax.set_xticklabels(['Not Placed','Placed'], rotation=0)
                    st.pyplot(fig)

                    # Salary distribution
                    if 'Predicted_Salary' in df.columns:
                        fig2, ax2 = plt.subplots()
                        df['Predicted_Salary'].plot(kind='hist', bins=12, color='skyblue', ax=ax2)
                        ax2.set_title("Predicted Salary Distribution")
                        ax2.set_xlabel("Salary (LPA)")
                        st.pyplot(fig2)

                    # Branch-wise average salary (if Branch exists)
                    if 'Branch' in df.columns:
                        # If Branch is encoded numeric, try to map back if encoder available
                        branch_col = df['Branch']
                        try:
                            # If branch values are strings, use them directly
                            if branch_col.dtype == object:
                                branch_names = branch_col
                            else:
                                # Map numeric codes back to labels if encoder exists
                                if le_branch is not None:
                                    classes = list(le_branch.classes_)
                                    branch_names = branch_col.map(lambda x: classes[x] if 0 <= int(x) < len(classes) else f"Code_{x}")
                                else:
                                    branch_names = branch_col.astype(str)
                        except Exception:
                            branch_names = branch_col.astype(str)

                        df['Branch_Name'] = branch_names
                        branch_salary = df.groupby('Branch_Name')['Predicted_Salary'].mean().sort_values(ascending=False)
                        st.subheader("Average Predicted Salary by Branch")
                        st.bar_chart(branch_salary)

                    # Interactive filters
                    st.subheader("Interactive Filters")
                    if 'Branch_Name' in df.columns:
                        branch_choice = st.selectbox("Filter by Branch (All shows all)", options=["All"] + sorted(df['Branch_Name'].unique().tolist()))
                        if branch_choice != "All":
                            filtered = df[df['Branch_Name'] == branch_choice]
                        else:
                            filtered = df
                    else:
                        filtered = df

                    if 'Predicted_Salary' in filtered.columns:
                        st.write(f"Showing {len(filtered)} records after filter.")
                        st.dataframe(filtered.head(100))
                        st.subheader("Filtered Salary Distribution")
                        st.bar_chart(filtered['Predicted_Salary'].reset_index(drop=True))

            except Exception as e:
                st.error(f"⚠️ Error processing file: {e}")

        else:
            st.info("Upload a CSV file containing student records. Expected columns include CGPA, Branch, Skills/Aptitude/Communication or similar. The app will attempt to preprocess automatically.")

       # -------------------------------
    # Tab 3: Resume Builder & ATS Analyzer
    # -------------------------------
    with tab3:
        st.header("📄 Resume Builder & ATS Analyzer")

        uploaded_resume = st.file_uploader("Upload your resume (TXT or PDF)", type=["txt", "pdf"])
        if uploaded_resume is not None:
            try:
                # Extract text from resume
                resume_text = ""
                if uploaded_resume.type == "application/pdf":
                    import PyPDF2
                    reader = PyPDF2.PdfReader(uploaded_resume)
                    for page in reader.pages:
                        resume_text += page.extract_text() + "\n"
                else:
                    resume_text = uploaded_resume.read().decode("utf-8")

                st.subheader("Resume Preview")
                st.text_area("Extracted Resume Text", resume_text, height=200)

                # ATS keyword list (expandable based on job trends)
                ats_keywords = [
                    "Python", "SQL", "Machine Learning", "Data Analysis", "Communication",
                    "Internship", "Project", "Cloud", "Cybersecurity", "Leadership"
                ]
                missing_keywords = [kw for kw in ats_keywords if kw.lower() not in resume_text.lower()]
                present_keywords = [kw for kw in ats_keywords if kw.lower() in resume_text.lower()]

                # Resume scoring system
                score = 100
                if missing_keywords:
                    score -= len(missing_keywords) * 5
                word_count = len(resume_text.split())
                if word_count < 150:
                    score -= 10
                elif word_count > 600:
                    score -= 10

                st.subheader("ATS Resume Score")
                st.progress(score / 100)
                st.info(f"Your resume scored {score}/100 based on ATS checks.")

                # Keyword coverage visualization
                coverage_data = pd.DataFrame({
                    "Keywords": ["Present", "Missing"],
                    "Count": [len(present_keywords), len(missing_keywords)]
                })
                st.subheader("Keyword Coverage")
                st.bar_chart(coverage_data.set_index("Keywords"))

                # Feedback on keywords
                if missing_keywords:
                    st.error(f"⚠️ Missing important keywords: {', '.join(missing_keywords)}")
                else:
                    st.success("✅ Your resume contains all the essential ATS keywords!")

                # Formatting feedback
                if word_count < 150:
                    st.warning("Your resume seems too short. Add more details about projects and achievements.")
                elif word_count > 600:
                    st.warning("Your resume seems too long. Keep it concise (1–2 pages).")
                else:
                    st.info("Resume length looks good.")

                # Suggestions with examples
                st.subheader("Suggestions for Improvement")
                st.write("🔹 **Problem: Missing Keywords** → Example Fix: If 'Machine Learning' is missing, add a project line like: *'Built a machine learning model to predict student placements with 85% accuracy.'*")
                st.write("🔹 **Problem: Too Short** → Example Fix: Expand project descriptions: *'Developed a web app using Python and SQL to manage student records, improving efficiency by 30%.'*")
                st.write("🔹 **Problem: Too Long** → Example Fix: Remove filler lines: Instead of *'I am a hardworking student'*, write *'Completed 3 internships in data analytics and cloud computing.'*")
                st.write("🔹 **Problem: Weak Formatting** → Example Fix: Use bullet points:\n- Internship at XYZ Corp (2025)\n- Built predictive model in Python\n- Improved accuracy by 15%")
                st.write("🔹 **Problem: Missing Achievements** → Example Fix: Add measurable results: *'Led a team of 4 to develop a cybersecurity dashboard, reducing incident response time by 20%.'*")

            except Exception as e:
                st.error(f"⚠️ Error analyzing resume: {e}")
        else:
            st.info("Upload a resume file to get instant ATS feedback and improvement suggestions.")

else:
    st.info("Please login or register from the sidebar to access the app.")
