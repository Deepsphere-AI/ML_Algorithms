import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import source.title_1 as head
# import title_1 as head

def Anomaly():
    head.title()
    st.markdown("<p style='text-align: center; color: black; font-size:20px;'><span style='font-weight: bold'>Problem Statement: </span>Application to Detect anomaly users in social media</p>", unsafe_allow_html=True)
    st.markdown("<hr style=height:2.5px;background-color:gray>",unsafe_allow_html=True)
    preview = None
    df = None
    train_button = False
    vAR_model = None
    csv_file = None
    csv_file_test = None
    visual = None
    df_test = pd.DataFrame({})

    col11,col12,col13,col14,col15 = st.columns([1.5,4,4.75,1,1.75])
    with col11:
        st.write("")
    with col12:
        # st.write("# ")
        st.write("# ")
        st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Problem Statement </span></p>", unsafe_allow_html=True)
    with col13:
        vAR_problem = st.selectbox("",["Select Problem Statement","Detect the anomaly users"])
    with col14:
        st.write("")
    with col15:
        st.write("")
    
    with col11:
        st.write("")
    with col12:
        if vAR_problem=="Detect the anomaly users":
            st.write("# ")
            st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Model Selection</span></p>", unsafe_allow_html=True)
    with col13:
        if vAR_problem=="Detect the anomaly users":
            vAR_model = st.selectbox("",["Select","Isolation Forest"])
    with col14:
        st.write("")
    with col15:
        st.write("")
    
    col21,col22,col23,col24,col25 = st.columns([1.5,4,4.75,1,1.75])

    with col21:
        st.write("")
    with col22:
        if vAR_model =="Isolation Forest":
                st.write("# ")
                st.write("")
                st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Upload Train Data </span></p>", unsafe_allow_html=True)
    with col23:
        if vAR_model =="Isolation Forest":
            csv_file = st.file_uploader("",type="csv",key='Train')
            if csv_file != None:
                df = pd.read_csv(csv_file)
                features = ["login_activity", "posting_activity", "social_connections"]
                X = df[features]
                model = IsolationForest(n_estimators=100, contamination=0.1)
                model.fit(X)
                y_pred = model.predict(X)
    with col24:
        st.write("")      
    with col25:
        if csv_file != None:
            st.write("# ")
            st.write("")
            preview = st.button("Preview")
    
    if preview == True:
        st.table(df.head(10))

    col31,col32,col33,col34,col35 = st.columns([1.5,4,4.75,1,1.75])
    with col31:
        st.write("")
    with col32:
        st.write("")       
    with col33:
        if csv_file != None:
            train_button = st.button("Train the model")
        if train_button == True:
            st.success("Model Training is successful")
    with col34:
        st.write("")
    with col35:
        st.write("")

    col41,col42,col43,col44,col45 = st.columns([1.5,4,4.75,1,1.75])
    with col41:
        st.write("")
    with col42: 
        if csv_file != None:
            st.write("#")
            st.write("")
            st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Upload Test Data </span></p>", unsafe_allow_html=True)
    with col43:
        if csv_file!=None:
            csv_file_test = st.file_uploader("",type="csv",key='test')

        if csv_file_test != None:
            df_test = pd.read_csv(csv_file_test)
            x=df_test[["login_activity", "posting_activity", "social_connections"]]
            df_values=x.values
            find=df_values
            result=[]
            for i in find:
                z=model.predict([i])
                if z==[1]:
                    result.append('no')
                elif z==[-1]:
                    result.append('yes')
            df_test['Anomaly']=result
    with col44:
        st.write("")
    with col45:
        if csv_file_test != None:
            st.write("# ")
            st.write("")
            preview = st.button("Predict",key="preview2")
    if preview == True:
        st.table(df_test.head())

    
    col51,col52,col53,col54,col55 = st.columns([1.5,4,4.75,1,1.75])
    with col51:
        st.write("")
    with col52:
        st.write("")
    with col52:
        if csv_file_test != None:
            st.write("")
            st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Visualization</span></p>", unsafe_allow_html=True)
    with col53:
        if csv_file_test != None:
            visual=st.selectbox("",["Select","Scatter"])
        if visual=="Scatter":
            df["anomaly_score"] = model.decision_function(X)
            anomalies = df.loc[df["anomaly_score"] < 0]
            fig,ax = plt.subplots(1,1,figsize=(15,10))
            ax.scatter(df["social_connections"], df["anomaly_score"], label="Normal")
            ax.scatter(anomalies["social_connections"], anomalies["anomaly_score"], color="r", label="Anomaly")
            ax.set_xlabel("Social Connections")
            ax.set_ylabel("anomaly_score")
            ax.legend()
            st.pyplot(fig)

    with col54:
        st.write()
    with col55:
        st.write()
                    
                    
# Anomaly()
