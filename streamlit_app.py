import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Superplasticizer Dosage Predictor",
    page_icon="🧪",
    layout="wide"
)

# Load models
@st.cache_resource
def load_models():
    """Load trained model and encoders"""
    model = joblib.load('optimal_dosage_predictor.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    saturation_data = pd.read_csv('saturation_points.csv')
    flow_data = pd.read_csv('processed_flow_data.csv')
    return model, label_encoder, saturation_data, flow_data

try:
    model, le, saturation_df, flow_df = load_models()
    models_loaded = True
except Exception as e:
    models_loaded = False
    st.error(f"⚠️ Error loading models: {e}")

# Title and description
st.title("🧪 Optimal Superplasticizer Dosage Predictor")
st.markdown("""
This tool predicts the optimal dosage of superplasticizer (PCE or SNF) for cement paste 
based on the **Marsh Cone Test** results. The optimal dosage represents the **saturation point** 
where adding more superplasticizer does not improve fluidity.
""")

st.divider()

if models_loaded:
    # Create two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📋 Input Parameters")
        
        # User inputs
        wc_ratio = st.number_input(
            "Water-to-Cement Ratio (W/C)",
            min_value=0.30,
            max_value=0.50,
            value=0.40,
            step=0.01,
            format="%.2f",
            help="Typical range: 0.35 - 0.45 for high-strength concrete"
        )
        
        sp_type = st.selectbox(
            "Superplasticizer Type",
            options=['PCE', 'SNF'],
            help="PCE = Polycarboxylate Ether, SNF = Sulfonated Naphthalene Formaldehyde"
        )
        
        # Silica Fume
        sf_pct = st.number_input(
            "Silica Fume Percentage (%)",
            min_value=0.0,
            max_value=20.0,
            value=0.0,
            step=1.0,
            format="%.1f",
            help="Percentage of Silica Fume replacement (e.g., 0, 5, 10, 15)"
        )
        sf_cg = st.number_input(
            "Cement Grade",
            min_value=0.0,
            max_value=100.0,
            value=53.0,
            step=1.0,
            format="%.1f",
            help="Cement Grade Value (e.g., 53)"
        )
        
        # Predict button
        if st.button("🔮 Predict Optimal Dosage", type="primary"):
            # Encode and predict
            sp_encoded = le.transform([sp_type])[0]
            # Feature order: [wc_ratio, sp_type, silica_fume]
            X = np.array([[wc_ratio, sp_encoded, sf_pct]])
            prediction = model.predict(X)[0]
            
            # Store in session state
            st.session_state['prediction'] = prediction
            st.session_state['wc_ratio'] = wc_ratio
            st.session_state['sp_type'] = sp_type
            st.session_state['sf_pct'] = sf_pct
    
    with col2:
        st.subheader("🎯 Prediction Results")
        
        if 'prediction' in st.session_state:
            pred = st.session_state['prediction']
            wc = st.session_state['wc_ratio']
            sp = st.session_state['sp_type']
            sf = st.session_state.get('sf_pct', 0)
            
            # Display prediction in a nice metric card
            st.metric(
                label=f"Optimal {sp} Dosage",
                value=f"{pred:.3f}%",
                help="This is the saturation point dosage"
            )
            
            # Additional information
            st.success("✅ Prediction Complete!")
            
            with st.expander("📊 Interpretation & Guidelines", expanded=True):
                st.markdown(f"""
                **Mix Details:**
                - W/C Ratio: `{wc:.2f}`
                - Silica Fume: `{sf}%`
                
                **Recommended Dosage:** `{pred:.3f}%` of binder weight
                
                **What this means:**
                - For 100 kg of binder (cement + SF), use **{pred*1000:.1f} grams** of {sp}
                - This is the **saturation point** prediction
                - Adding more than this amount will **not improve** fluidity
                - May cause segregation or bleeding if exceeded
                
                **Practical Application:**
                - Use this dosage for your concrete mix design
                - Adjust slightly based on field conditions
                - Monitor slump flow and workability
                """)
            
            # Show comparison with training data
            st.markdown("### 📈 Predicted vs Actual (Exact Match)")
            
            # Find exact match in saturation points data
            match = saturation_df[
                (saturation_df['sp_type'] == sp) & 
                (np.isclose(saturation_df['wc_ratio'], wc)) & 
                (np.isclose(saturation_df['silica_fume'], sf))
            ]
            
            if not match.empty:
                actual_dosage = match.iloc[0]['optimal_dosage']
                
                comparison_df = pd.DataFrame({
                    'Type': ['Predicted', 'Actual'],
                    'Dosage (%)': [pred, actual_dosage]
                })

                fig = px.bar(
                    comparison_df,
                    x='Type',
                    y='Dosage (%)',
                    color='Type',
                    title=f'{sp} Dosage Comparison (W/C {wc}, SF {sf}%)',
                    color_discrete_map={'Predicted': '#FF6B6B', 'Actual': '#4ECDC4'},
                    text='Dosage (%)'
                )
                
                fig.update_traces(texttemplate='%{text:.3f}%', textposition='outside')
                fig.update_layout(yaxis_title="Optimal Dosage (%)", showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # Show Error
                error = pred - actual_dosage
                st.info(f"Difference (Pred - Act): **{error:+.4f}%**")
            else:
                st.info(f"ℹ️ No exact actual data point found for **W/C {wc:.2f}** and **Silica Fume {sf}%** in the historical dataset.")
        else:
            st.info("👈 Enter parameters and click 'Predict' to see results")
    
    # Visualizations section
    st.divider()
    st.subheader("📊 Flow Curve Analysis")
    
    tab1, tab2, tab3 = st.tabs(["Flow Curves", "Saturation Points", "Data Table"])
    
    with tab1:
        st.markdown("**Flow time vs. Superplasticizer dosage for different W/C ratios**")
        
        col_a, col_b = st.columns(2)
        with col_a:
            selected_sp = st.radio("Select Superplasticizer", ['PCE', 'SNF'], horizontal=True, key='flow_curve_sp')
        with col_b:
            selected_sf_flow = st.select_slider("Select Silica Fume %", options=[0, 5, 10, 15], key='flow_curve_sf')
        
        # Filter data
        filtered_data = flow_df[(flow_df['sp_type'] == selected_sp) & 
                               (flow_df['silica_fume'] == selected_sf_flow)]
        
        # Create plot
        fig = go.Figure()
        
        colors = {0.35: '#E63946', 0.40: '#F77F00', 0.45: '#06A77D'}
        
        for wc in [0.35, 0.40, 0.45]:
            subset = filtered_data[filtered_data['wc_ratio'] == wc]
            fig.add_trace(go.Scatter(
                x=subset['dosage'],
                y=subset['flow_time'],
                mode='lines+markers',
                name=f'W/C {wc}',
                line=dict(width=3, color=colors[wc]),
                marker=dict(size=10)
            ))
            
            # Add saturation point marker
            sat_point = saturation_df[(saturation_df['sp_type'] == selected_sp) & 
                                     (saturation_df['wc_ratio'] == wc) &
                                     (saturation_df['silica_fume'] == selected_sf_flow)]
            if not sat_point.empty:
                fig.add_trace(go.Scatter(
                    x=[sat_point['optimal_dosage'].values[0]],
                    y=[sat_point['min_flow_time'].values[0]],
                    mode='markers',
                    name=f'Saturation (W/C {wc})',
                    marker=dict(size=15, symbol='star', color=colors[wc], 
                              line=dict(width=2, color='black')),
                    showlegend=False
                ))
        
        fig.update_layout(
            title={
                'text': f'{selected_sp} Flow Curves (⭐ = Saturation Point)',
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title='Dosage (%)',
            yaxis_title='Flow Time (seconds)',
            hovermode='x unified',
            height=500,
            margin=dict(l=20, r=20, t=60, b=20)
        )
        
        # Center the chart using columns
        c1, c2, c3 = st.columns([1, 10, 1])
        with c2:
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("**Comparison of saturation dosages vs Silica Fume content**")
        
        # Saturation comparison
        fig = px.bar(
            saturation_df,
            x='silica_fume',
            y='optimal_dosage',
            color='sp_type',
            barmode='group',
            facet_col='wc_ratio',
            title='Saturation Dosage vs Silica Fume % (by W/C Ratio)',
            labels={'wc_ratio': 'W/C', 'optimal_dosage': 'Optimal Dosage (%)', 'sp_type': 'SP Type', 'silica_fume': 'Silica Fume %'},
            color_discrete_map={'PCE': '#2E86AB', 'SNF': '#A23B72'},
            text='optimal_dosage'
        )
        
        fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig.update_layout(height=500)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**PCE Summary**")
            pce_stats = saturation_df[saturation_df['sp_type'] == 'PCE']['optimal_dosage']
            st.metric("Average Dosage", f"{pce_stats.mean():.3f}%")
            st.metric("Range", f"{pce_stats.min():.2f}% - {pce_stats.max():.2f}%")
        
        with col2:
            st.markdown("**SNF Summary**")
            snf_stats = saturation_df[saturation_df['sp_type'] == 'SNF']['optimal_dosage']
            st.metric("Average Dosage", f"{snf_stats.mean():.3f}%")
            st.metric("Range", f"{snf_stats.min():.2f}% - {snf_stats.max():.2f}%")
    
    with tab3:
        st.markdown("**Complete saturation points dataset**")
        st.dataframe(
            saturation_df.style.format({
                'wc_ratio': '{:.2f}',
                'optimal_dosage': '{:.3f}',
                'min_flow_time': '{:.1f}'
            }),
            use_container_width=True
        )
        
        # Download button
        csv = saturation_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Data (CSV)",
            data=csv,
            file_name="saturation_points.csv",
            mime="text/csv"
        )

    # Model information
    st.divider()
    with st.expander("ℹ️ About the Model"):
        st.markdown("""
        **Model Details:**
        - **Algorithm:** Gradient Boosting Regressor
        - **Training Data:** 6 saturation points (3 W/C ratios × 2 SP types)
        - **Performance:** MAE = 0.025%, RMSE = 0.035%
        - **Validation:** Leave-One-Out Cross-Validation
        
        **Input Features:**
        1. Water-to-Cement Ratio (0.35 - 0.45)
        2. Superplasticizer Type (PCE or SNF)
        3. Silica Fume % (0 - 15)   
        
        **Output:**
        - Optimal dosage at saturation point (%)
        """)

else:
    st.warning("Please ensure all model files are present in the working directory.")
