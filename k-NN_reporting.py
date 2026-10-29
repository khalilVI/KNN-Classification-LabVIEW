import streamlit as st
import pandas as pd

data = {
    'rid': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    'age': ['<=30', '<=30', '31...40', '>40', '>40', '>40', '31...40', '<=30', '<=30', '>40', '<=30', '31...40', '31...40', '>40'],
    'income': ['high', 'high', 'high', 'medium', 'low', 'low', 'low', 'medium', 'low', 'medium', 'medium', 'medium', 'high', 'medium'],
    'student': ['no', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no'],
    'credit_rating': ['fair', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'excellent'],
    'buys_computer': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']
}

df = pd.DataFrame(data)

def classify_customer(d_new, k_value):
    weights = {'age': 1.0, 'income': 2.0, 'student': 1.0, 'credit_rating': 1.0}
    results = []

    for index, row in df.iterrows():
        current_score = 0.0
        breakdown = [] 
        
        if row['age'] == d_new['age']:
            current_score += weights['age']
            breakdown.append("1")
        else:
            breakdown.append("0")

        if row['income'] == d_new['income']:
            current_score += weights['income']
            breakdown.append("2")
        else:
            breakdown.append("0")

        if row['student'] == d_new['student']:
            current_score += weights['student']
            breakdown.append("1")
        else:
            breakdown.append("0")

        if row['credit_rating'] == d_new['credit_rating']:
            current_score += weights['credit_rating']
            breakdown.append("1")
        else:
            breakdown.append("0")
            
        final_sim = current_score / 4.0 
        
        results.append({
            'RID': row['rid'],
            'Math': "+".join(breakdown),
            'Similarity': final_sim,
            'Class': row['buys_computer']
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='Similarity', ascending=False)
    top_k_neighbors = results_df.head(k_value)
    
    prediction = top_k_neighbors['Class'].mode()[0]
    
    return prediction, top_k_neighbors

st.set_page_config(page_title="Training Data", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background-color: #191E29;
        color: #FFFFFF;
    }
    
    [data-testid="stSidebar"] {
        background-color: #132D46;
    }
    
    .stSelectbox label, .stSlider label {
        color: #01C38D !important;
        font-weight: bold;
        font-size: 16px;
    }
    
    div.stButton > button {
        background-color: #01C38D;
        color: #191E29;
        border: none;
        font-weight: bold;
        font-size: 18px;
        padding: 12px;
        border-radius: 8px;
        width: 100%;
        transition: all 0.3s;
    }
    div.stButton > button:hover {
        background-color: #FFFFFF;
        color: #01C38D;
        box-shadow: 0 0 10px #01C38D;
    }
    
    .name-box {
        position: fixed;
        top: 60px;
        right: 30px;
        background-color: #132D46;
        padding: 15px 25px;
        border-radius: 12px;
        border: 2px solid #01C38D;
        box-shadow: 0px 4px 15px rgba(1, 195, 141, 0.2);
        z-index: 9999;
        text-align: center;
        width: auto;
    }
    
    h1, h2, h3 {
        color: #FFFFFF !important;
        font-family: 'TT Commons Demibold', sans-serif;
    }
    
    [data-testid="stDataFrame"] {
        border: 1px solid #696E79;
        border-radius: 5px;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    
    <div class="name-box">
        <span style="color: #696E79; font-size: 12px; text-transform: uppercase; letter-spacing: 1px;">Project By</span><br>
        <span style="font-size: 20px; color: #FFFFFF; font-weight: bold;">KHALIL ELYOUMNI</span>
    </div>
    """, unsafe_allow_html=True)

with st.sidebar:
    st.header("Training Data Table")
    st.dataframe(df, height=600)

st.title("Weighted KNN Classifier")

st.header("Configuration")
k = st.slider("Select K (Neighbors)", min_value=3, max_value=5, value=5)

st.divider()

st.header("Define New Customer")
col1, col2 = st.columns(2)

with col1:
    age_input = st.selectbox("Age Group", options=['<=30', '31...40', '>40'], key="age")
    income_input = st.selectbox("Income Level", options=['high', 'medium', 'low'], key="income")

with col2:
    student_input = st.selectbox("Is Student?", options=['yes', 'no'], key="student")
    credit_input = st.selectbox("Credit Rating", options=['fair', 'excellent'], key="credit")

d_new_user = {
    'age': age_input,
    'income': income_input,
    'student': student_input,
    'credit_rating': credit_input
}

st.write("") 

if st.button("Classify Customer", type="primary"):
    prediction, details_df = classify_customer(d_new_user, k)
    
    st.divider()
    
    if prediction == 'yes':
        st.markdown(f"""
            <div style="background-color: #01C38D; padding: 20px; border-radius: 10px; text-align: center;">
                <h2 style="color: #191E29 !important; margin:0;">✅ PREDICTION: YES (Buys Computer)</h2>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div style="background-color: #D32F2F; padding: 20px; border-radius: 10px; text-align: center;">
                <h2 style="color: #FFFFFF !important; margin:0;">❌ PREDICTION: NO (Does not buy)</h2>
            </div>
            """, unsafe_allow_html=True)
        
    st.write("")
    st.subheader(f"Top {k} Nearest Neighbors Analysis")
    
    details_df['Matches (1+2+1+1)'] = details_df['Similarity'].apply(lambda x: f"{int(round(x * 4))}/4")
    details_df = details_df[['Math', 'Matches (1+2+1+1)', 'Similarity', 'Class']]
    
    st.dataframe(details_df, use_container_width=True)