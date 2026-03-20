import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(
    page_title="社区慢病老年人衰弱风险计算器",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load model and data
OPTIMAL_THRESHOLD = 0.204
model = joblib.load('CatBoost_frailty1.pkl')
scaler = joblib.load('scaler frailty1.pkl') 

# Define options
每日食用蔬菜的量_options = {    
    0: '＜300g',    
    1: '300-500g',    
    2: '＞500g',    
}

认知状态_options = {       
    0: '认知正常',    
    1: '轻度认知障碍',    
    2: '中度认知障碍',
    3: '中度认知障碍',
}

# Feature names
feature_names = [
    "年龄", "用药数量", "每日食用蔬菜的量", "血红蛋白浓度",  "认知状态"
]

# Custom CSS for compact layout
st.markdown("""
    <style>
    /* Main styling */
    .main {
        background-color: #f8f9fa;
    }
    .st-bw {
        background-color: white;
    }
    
    /* Form styling */
    .stNumberInput, .stSelectbox {
        padding-bottom: 4px;
    }
    div[data-baseweb="input"] {
        margin-bottom: -1rem;
    }
    
    /* Right column styling */
    .right-column {
        font-size: 0.9rem;
    }
    .right-title {
        text-align: left;
        margin-top: 0;
        padding-top: 0;
        font-size: 1.3rem;  /* Reduced from 1.5rem */
    }
    
    /* Prediction box */
    .prediction-box {
        border-radius: 5px;
        padding: 12px;
        margin-bottom: 12px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .high-risk {
        background-color: #ffdddd;
        border-left: 4px solid #ff5252;
    }
    .low-risk {
        background-color: #ddffdd;
        border-left: 4px solid #4caf50;
    }
    
    /* Equal height columns */
    .st-emotion-cache-1cypcdb {
        align-items: stretch;
    }
    
    /* Section headers */
    .section-header {
        font-size: 0.95rem;
        font-weight: bold;
        margin-bottom: 8px;
    }
    
    /* Make columns equal height */
    .column-container {
        display: flex;
        flex-direction: row;
    }
    
    /* Abbreviations style */
    .abbreviations {
        font-size: 0.8rem;
        margin-top: 10px;
        color: #555;
    }
    </style>
    """, unsafe_allow_html=True)

# Create two columns (40%, 60%)
col1, col2 = st.columns([4, 6], gap="medium")

# Right column content
with col2:
    st.markdown("<div class='right-column'>", unsafe_allow_html=True)
    
    # Title and description in right column (with smaller font)
    #st.markdown("<h1 style='font-size:1.3rem'>社区慢病老年人衰弱风险计算器</h1>", unsafe_allow_html=True)
    st.markdown("<h1 class='right-title'>社区慢病老年人衰弱风险计算器</h1>", unsafe_allow_html=True)
    #st.markdown("<p>This tool predicts the risk of frailty in heart failure patients with acute infections.</p>", unsafe_allow_html=True)

# Left column content - input form
with col1:
    with st.container():
        with st.form("input_form"):
            # Demographic Information
            年龄 = st.number_input("年龄", min_value=1, max_value=150, value=60)
            用药数量 = st.number_input("用药数量", min_value=1, max_value=30, value=3)
            
            # Clinical Characteristics
            每日食用蔬菜的量 = st.selectbox(
                "每日食用蔬菜的量", 
                options=list(每日食用蔬菜的量_options.keys()), 
                format_func=lambda x: 每日食用蔬菜的量_options[x]
            )
            # Laboratory Values
            血红蛋白浓度 = st.number_input(
                "血红蛋白浓度 (g/L)", 
                min_value=0.0, max_value=500.0, value=150.0, step=1.0, format="%.1f"
            )
                        
            submitted = st.form_submit_button("预测", use_container_width=True)

# Prepare input features and show results when submitted
if submitted:
    feature_values = [
        年龄, 用药数量, 每日食用蔬菜的量, 血红蛋白浓度, 认知状态
        ]
    
    features = np.array([feature_values])
    
    # Data preprocessing
    continuous_features = [年龄, 用药数量,血红蛋白浓度]
    categorical_features = [每日食用蔬菜的量,认知状态]
    
    continuous_features_df = pd.DataFrame(
        np.array(continuous_features).reshape(1, -1), 
        columns=["年龄", "用药数量", "血红蛋白浓度"]
    )
    
    continuous_features_standardized = scaler.transform(continuous_features_df)
    categorical_features_array = np.array(categorical_features).reshape(1, -1)
    final_features = np.hstack([continuous_features_standardized, categorical_features_array])
    final_features_df = pd.DataFrame(final_features, columns=feature_names)
    
    # Prediction
    predicted_proba = model.predict_proba(final_features_df)[0]
    prob_class1 = predicted_proba[1]
    predicted_class = 1 if prob_class1 >= OPTIMAL_THRESHOLD else 0
    
    with col2:
        # Prediction results
        risk_class = "high-risk" if predicted_class == 1 else "low-risk"
        st.markdown(
            f"""
            <div class="prediction-box {risk_class}">
                <h3 style='margin-top:0; font-size: 1.1rem;'>预测结果</h3>
                <p style="font-size:1rem; font-weight:bold; margin-bottom:0;">
                    Frailty Probability: <span style="color:{'#ff5252' if predicted_class == 1 else '#4caf50'}">{prob_class1:.1%}</span>
                </p>
                <p style="font-size:0.9rem;">
                    Risk Classification: <strong>{'High Risk' if predicted_class == 1 else 'Low Risk'}</strong>
                    (Threshold: {OPTIMAL_THRESHOLD:.0%})
                </p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # SHAP explanation plot
        st.markdown("<div class='section-header'>Feature Impact Analysis</div>", unsafe_allow_html=True)
        with st.spinner("Generating explanation..."):
            explainer_shap = shap.TreeExplainer(model)
            shap_values = explainer_shap.shap_values(final_features_df)
            
            if isinstance(shap_values, list):
                shap_values_class = shap_values[0]
            else:
                shap_values_class = shap_values
            
            original_feature_values = pd.DataFrame(
                features, 
                columns=feature_names
            )
            
            # Create SHAP plot with all features
            fig, ax = plt.subplots(figsize=(10, 8))  # 增加图形高度以适应更多特征
            shap.plots.waterfall(
                shap.Explanation(
                    values=shap_values_class[0], 
                    base_values=explainer_shap.expected_value,
                    data=original_feature_values.iloc[0],
                    feature_names=original_feature_values.columns.tolist()
                ),
                max_display=len(feature_names),  # 显示所有特征
                show=False
            )
            plt.title("Feature Contribution to Prediction", fontsize=12, pad=10)
            plt.gcf().set_size_inches(9, len(feature_names)*0.6)  # 动态调整高度
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            
            st.caption("""
            此图显示了每个特征对预测的贡献，红色特征会增加风险，蓝色特征则降低风险
            """)
            
            # Added abbreviations section
            #st.markdown("""
           # <div class='abbreviations'>
                #<strong>Abbreviations:</strong><br>
               # MCHC: Mean Corpuscular Hemoglobin Concentration<br>
                #eGFR: Estimated Glomerular Filtration Rate<br>
                #LVEF: Left Ventricular Ejection Fraction
            ##</div>
           # """, unsafe_allow_html=True)
        
        # Close right column div
        st.markdown("</div>", unsafe_allow_html=True)
