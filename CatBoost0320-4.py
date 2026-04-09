# --- 1. 修改列比例 (从 [4, 6] 改为 [3, 7]) ---
col1, col2 = st.columns([3, 7], gap="large")

# --- 2. 优化 CSS 样式，让输入框更紧凑 ---
st.markdown("""
    <style>
    /* 减少表单内部的间距 */
    [data-testid="stForm"] {
        padding: 1.5rem !important; /* 缩小内边距 */
    }
    
    /* 让数字输入框和下拉框更矮一点，节省空间 */
    .stNumberInput div, .stSelectbox div {
        margin-bottom: -5px;
    }
    
    /* 标题稍微缩小一点以适应窄列 */
    label[data-testid="stWidgetLabel"] p {
        font-size: 0.95rem !important;
        font-weight: 600 !important;
    }

    /* 结果区域的样式微调 */
    .result-container p {
        margin-bottom: 8px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. 布局逻辑 (保持不变，应用新的比例) ---
with col1:
    with st.form("input_form"):
        # 输入组件...
        age = st.number_input("年龄", 1, 150, 70)
        cog = st.selectbox("认知状态", options=list(cog_options.keys()), format_func=lambda x: cog_options[x], index=2)
        meds = st.number_input("用药数量", 1, 30, 5)
        veg = st.selectbox("每日食用蔬菜的量", options=list(veg_options.keys()), format_func=lambda x: veg_options[x])
        hemo = st.number_input("血红蛋白浓度 (g/L)", 0.0, 500.0, 150.0, step=1.0, format="%.1f")
        
        submitted = st.form_submit_button("预测", use_container_width=True)

with col2:
    if submitted:
        # 结果和 SHAP 图表逻辑...
        # 提示：由于 col2 变宽了，你可以适当增加绘图的高度
        fig, ax = plt.subplots(figsize=(10, 4.5)) 
        # ... 绘图代码 ...