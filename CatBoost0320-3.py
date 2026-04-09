import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# --- 页面配置 ---
st.set_page_config(
    page_title="社区慢病老年人衰弱风险计算器",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 加载模型和数据 (保持你的原逻辑) ---
@st.cache_resource
def load_assets():
    model = joblib.load('CatBoost_frailty1.pkl')
    scaler = joblib.load('scaler frailty1.pkl')
    return model, scaler

model, scaler = load_assets()
OPTIMAL_THRESHOLD = 0.204
feature_names_zh = ["年龄", "用药数量", "每日食用蔬菜的量", "血红蛋白浓度", "认知状态"]
feature_names_en = ["Age", "Medication Count", "Vegetable Intake", "Hemoglobin", "Cognitive Status"]

# --- 选项定义 ---
veg_options = {0: '＜300g', 1: '300-500g', 2: '＞500g'}
cog_options = {0: '认知正常', 1: '轻度认知障碍', 2: '中度认知障碍', 3: '重度认知障碍'}

# --- 核心 CSS 修改：实现对齐和紧凑感 ---
st.markdown("""
    <style>
    /* 全局背景和字体 */
    .main { background-color: #ffffff; }
    
    /* 标题样式 */
    .main-title {
        text-align: center;
        color: #31333F;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 2rem;
    }
    
    /* 左侧输入区域表单样式 */
    [data-testid="stForm"] {
        border: 1px solid #e6e9ef;
        border-radius: 10px;
        background-color: #f8f9fa;
        padding: 2rem;
    }
    
    /* 结果显示区域 */
    .result-container {
        padding-left: 20px;
    }
    .res-label { font-size: 1.2rem; font-weight: 500; display: inline-block; width: 150px; }
    .res-value { font-size: 1.2rem; font-weight: bold; }
    .risk-high { background-color: #ff4b4b; color: white; padding: 2px 10px; border-radius: 4px; }
    .risk-low { background-color: #28a745; color: white; padding: 2px 10px; border-radius: 4px; }
    
    /* 调整 SHAP 图表间距 */
    .shap-header { margin-top: 20px; font-weight: bold; color: #555; }
    </style>
    """, unsafe_allow_html=True)

# --- 页面内容 ---
st.markdown("<h1 class='main-title'>社区慢病老年人衰弱风险计算器</h1>", unsafe_allow_html=True)

col1, col2 = st.columns([4, 6], gap="large")

with col1:
    with st.form("input_form"):
        age = st.number_input("年龄", 1, 150, 70)
        cog = st.selectbox("认知状态", options=list(cog_options.keys()), format_func=lambda x: cog_options[x], index=2)
        meds = st.number_input("用药数量", 1, 30, 5)
        veg = st.selectbox("每日食用蔬菜的量", options=list(veg_options.keys()), format_func=lambda x: veg_options[x])
        hemo = st.number_input("血红蛋白浓度 (g/L)", 0.0, 500.0, 150.0, step=1.0, format="%.1f")
        
        submitted = st.form_submit_button("预测", use_container_width=True)

with col2:
    if submitted:
        # --- 预测逻辑 ---
        feature_values = [age, meds, veg, hemo, cog]
        # 标准化连续变量
        cont_features = pd.DataFrame([[age, meds, hemo]], columns=["年龄", "用药数量", "血红蛋白浓度"])
        cont_std = scaler.transform(cont_features)
        # 合并
        final_features = np.hstack([cont_std, np.array([[veg, cog]])])
        final_features_df = pd.DataFrame(final_features, columns=feature_names_zh)
        
        prob = model.predict_proba(final_features_df)[0][1]
        is_high = prob >= OPTIMAL_THRESHOLD
        
        # --- 结果展示 (完全匹配图片布局) ---
        st.markdown(f"""
            <div class='result-container'>
                <p><span class='res-label'>衰弱患病概率:</span> <span class='res-value' style='color:#ff4b4b;'>{prob:.1%}</span></p>
                <p><span class='res-label'>风险等级分层:</span> <span class='res-value risk-{"high" if is_high else "low"}'>{"高风险" if is_high else "低风险"}</span></p>
            </div>
        """, unsafe_allow_html=True)
        
        # --- SHAP 图表 ---
        st.markdown("<div class='shap-header'>Feature Impact Analysis (SHAP)</div>", unsafe_allow_html=True)
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(final_features_df)
        shap_v = shap_values[0] if isinstance(shap_values, list) else shap_values

        # 绘图美化
        fig, ax = plt.subplots(figsize=(8, 4)) # 调整尺寸以适应紧凑布局
        shap.plots.waterfall(
            shap.Explanation(
                values=shap_v[0] if len(shap_v.shape) > 1 else shap_v, 
                base_values=explainer.expected_value,
                data=np.array(feature_values),
                feature_names=feature_names_en
            ),
            show=False,
            max_display=5
        )
        plt.title("Contribution of each factor to risk", fontsize=9)
        st.pyplot(fig)
        
        st.caption("注：红色条柱表示该因素增加了衰弱风险，蓝色表示降低了风险。")
        
        # 对照表
        st.markdown("""
            <div style="font-size:0.8rem; color:#666; background-color:#f1f3f6; padding:12px; border-radius:8px; margin-top:10px;">
                <strong>指标翻译对照:</strong><br>
                Age: 年龄 | Medication Count: 用药数量 | Vegetable Intake: 蔬菜摄入量<br>
                Hemoglobin: 血红蛋白浓度 | Cognitive Status: 认知状态
            </div>
        """, unsafe_allow_html=True)
    else:
        # 未点击预测时的占位说明
        st.info("请在左侧输入数据并点击“预测”按钮查看分析结果。")