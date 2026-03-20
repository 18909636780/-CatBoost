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

# --- 关键修改：定义中英文特征名对照 ---
# 确保这个顺序与你 feature_values 列表中的顺序完全一致
feature_names_zh = ["年龄", "用药数量", "每日食用蔬菜的量", "血红蛋白浓度", "认知状态"]
feature_names_en = ["Age", "Medication Count", "Vegetable Intake", "Hemoglobin", "Cognitive Status"]

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

# Custom CSS
st.markdown("""
    <style>
    /* 新增：大标题居中样式 */
    .centered-title {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 20px;
        font-size: 2rem;
        font-weight: bold;
    }
    .main { background-color: #f8f9fa; }
    /* ... 其余 CSS 保持不变 ... */
    </style>
    """, unsafe_allow_html=True)

# 1. 关键位置：在创建列之前，先放置居中大标题
st.markdown("<h1 class='centered-title'>社区慢病老年人衰弱风险计算器</h1>", unsafe_allow_html=True)

# 2. 创建列容器 (现在 col2 里面不需要再放标题了)
col1, col2 = st.columns([4, 6], gap="medium")

# Right column content
with col2:
    st.markdown("<div class='right-column'>", unsafe_allow_html=True)
    # 这里删掉了原来的 st.markdown("<h1 class='right-title'>...</h1>")
    
# Left column content - input form
with col1:
    with st.container():
        with st.form("input_form"):
            年龄 = st.number_input("年龄", min_value=1, max_value=150, value=60)
            认知状态 = st.selectbox(
                "认知状态", 
                options=list(认知状态_options.keys()), 
                format_func=lambda x: 认知状态_options[x]
            )
            用药数量 = st.number_input("用药数量", min_value=1, max_value=30, value=3)
            每日食用蔬菜的量 = st.selectbox(
                "每日食用蔬菜的量", 
                options=list(每日食用蔬菜的量_options.keys()), 
                format_func=lambda x: 每日食用蔬菜的量_options[x]
            )
            血红蛋白浓度 = st.number_input(
                "血红蛋白浓度 (g/L)", 
                min_value=0.0, max_value=500.0, value=150.0, step=1.0, format="%.1f"
            )
            submitted = st.form_submit_button("预测", use_container_width=True)

# Prepare input features and show results when submitted
if submitted:
    # 按照 feature_names_zh 的顺序构建数值列表
    feature_values = [年龄, 用药数量, 每日食用蔬菜的量, 血红蛋白浓度, 认知状态]
    
    # 1. 连续变量标准化处理
    # 注意：这里的顺序必须和训练 scaler 时保持一致
    continuous_features_df = pd.DataFrame(
        [[年龄, 用药数量, 血红蛋白浓度]], 
        columns=["年龄", "用药数量", "血红蛋白浓度"]
    )
    continuous_features_standardized = scaler.transform(continuous_features_df)
    
    # 2. 合并分类变量
    categorical_features_array = np.array([[每日食用蔬菜的量, 认知状态]])
    final_features = np.hstack([continuous_features_standardized, categorical_features_array])
    
    # 3. 构造预测用的 DataFrame (必须带中文列名以适配模型)
    final_features_df = pd.DataFrame(final_features, columns=feature_names_zh)
    
    # Prediction
    predicted_proba = model.predict_proba(final_features_df)[0]
    prob_class1 = predicted_proba[1]
    predicted_class = 1 if prob_class1 >= OPTIMAL_THRESHOLD else 0
    
    with col2:
        # Prediction results
        risk_class = "high-risk" if predicted_class == 1 else "low-risk"
        risk_text = "高风险" if predicted_class == 1 else "低风险"
        risk_color = "#ff5252" if predicted_class == 1 else "#4caf50"
        
        st.markdown(
            f"""
            <div class="prediction-box {risk_class}">
                <h3 style='margin-top:0; font-size: 1.1rem; color: #333;'>评估结论</h3>
                <p style="font-size:1.1rem; margin-bottom:8px;">
                    衰弱患病概率: <span style="color:{risk_color}; font-weight:bold;">{prob_class1:.1%}</span>
                </p>
                <p style="font-size:1rem; margin-bottom:0;">
                    风险等级分层: <span style="background-color:{risk_color}; color:white; padding:2px 8px; border-radius:4px; font-weight:bold;">{risk_text}</span>
                </p>
               #<p style="font-size:0.8rem; color:#666; margin-top:10px; border-top:1px dashed #ccc; padding-top:5px;">
                    #注：诊断阈值为 {OPTIMAL_THRESHOLD:.1%}（概率超过此值即判定为高风险）
                #</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # SHAP explanation plot
        st.markdown("<div class='section-header'>Feature Impact Analysis (SHAP)</div>", unsafe_allow_html=True)
        with st.spinner("Generating explanation..."):
            explainer_shap = shap.TreeExplainer(model)
            shap_values = explainer_shap.shap_values(final_features_df)
            
            # 处理不同版本的 shap 输出格式
            shap_values_class = shap_values[0] if isinstance(shap_values, list) else shap_values
            
            # Create SHAP plot
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(
                shap.Explanation(
                    values=shap_values_class[0] if len(shap_values_class.shape) > 1 else shap_values_class, 
                    base_values=explainer_shap.expected_value,
                    data=np.array(feature_values), # 传入原始输入值
                    feature_names=feature_names_en  # 关键修改：绘图强制使用英文
                ),
                max_display=len(feature_names_en),
                show=False
            )
            plt.title("Contribution of each factor to risk", fontsize=10)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            
            st.caption("""
            **注：** 红色条柱表示该因素增加了衰弱风险，蓝色表示降低了风险。
            """)
            
            # 对照表帮助用户理解英文标签
            st.markdown("""
            <div style="font-size:0.8rem; color:#666; background-color:#f0f2f6; padding:10px; border-radius:5px;">
                <strong>指标翻译对照:</strong><br>
                Age: 年龄 | Medication Count: 用药数量 | Vegetable Intake: 蔬菜摄入量<br>
                Hemoglobin: 血红蛋白浓度 | Cognitive Status: 认知状态
            </div>
            """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
