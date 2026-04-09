import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 1. 必须先配置页面
st.set_page_config(
    page_title="社区慢病老年人衰弱风险计算器",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. 插入自定义 CSS 压缩间距
st.markdown("""
    <style>
    /* 进一步压缩左侧表单的内边距 */
    [data-testid="stForm"] {
        padding: 1rem !important;
        border-radius: 8px;
        background-color: #f8f9fa;
    }
    
    /* 紧凑型输入框设置 */
    .stNumberInput, .stSelectbox {
        margin-bottom: -15px;
    }

    /* 结果区域样式 */
    .res-container {
        padding-left: 10px;
        margin-top: -10px;
    }
    .risk-tag {
        background-color: #ff4b4b;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. 标题
st.markdown("<h2 style='text-align: center;'>社区慢病老年人衰弱风险计算器</h2>", unsafe_allow_html=True)
st.markdown("---")

# 4. 定义列比例：[2.5, 7.5] 让左边显著变窄
col1, col2 = st.columns([2.5, 7.5], gap="large")

with col1:
    with st.form("input_form"):
        st.markdown("**参数输入**")
        年龄 = st.number_input("年龄", 1, 150, 70)
        认知状态 = st.selectbox("认知状态", options=[0,1,2,3], format_func=lambda x: ['正常','轻度','中度','重度'][x], index=2)
        用药数量 = st.number_input("用药数量", 1, 30, 5)
        蔬菜量 = st.selectbox("每日蔬菜量", options=[0,1,2], format_func=lambda x: ['<300g','300-500g','>500g'][x])
        血红蛋白 = st.number_input("血红蛋白 (g/L)", 0.0, 500.0, 150.0)
        
        submitted = st.form_submit_button("开始预测", use_container_width=True)

with col2:
    if submitted:
        # 这里放置你的模型预测逻辑 (model.predict_proba 等)
        # 模拟数据用于演示布局
        prob = 0.67 
        
        # 结果对齐显示
        st.markdown(f"""
            <div class='res-container'>
                <p style='font-size:1.1rem;'>衰弱患病概率：<span style='color:#ff4b4b; font-weight:bold;'>{prob:.1%}</span></p>
                <p style='font-size:1.1rem;'>风险等级分层：<span class='risk-tag'>高风险</span></p>
            </div>
        """, unsafe_allow_html=True)

        # SHAP 图表
        st.write("### Feature Impact Analysis (SHAP)")
        # 增加 fig 的宽度比例，使其在宽列中更好看
        fig, ax = plt.subplots(figsize=(10, 4)) 
        # shap.plots.waterfall(...) 
        st.pyplot(fig)
    else:
        st.info("💡 请在左侧输入临床指标并点击预测。")
