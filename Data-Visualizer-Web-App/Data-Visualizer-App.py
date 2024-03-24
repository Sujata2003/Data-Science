# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 10:02:14 2024

@author: delll
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

#set page configuration
st.set_page_config(page_title="Data Visualizer",
                   layout="centered",
                   page_icon="📊")

# Title
st.title("📊 Data Visualizer - Web App")

#getting the working directory of the main.py
working_dir=os.path.dirname(os.path.abspath(__file__))

folder_path=f"{working_dir}/data"

files_list=[f for f in os.listdir(folder_path) if f.endswith(".csv")]

# drop down for all the files
selected_file=st.selectbox("Select a file",files_list,index=None)

if selected_file:
    # get complete path of the selected file
    file_path=os.path.join(folder_path,selected_file)
    
    #reading a csv file as a pandas dataframe
    df=pd.read_csv(file_path)
    
    col1,col2=st.columns(2)
    \
    columns=df.columns.tolist()
    with col1:
        st.write("")
        st.write(df.head())

    with col2:
        
        # user selection of df columns
        x_axis=st.selectbox("Select the x-axis",options=columns+["None"],index=None)
        y_axis=st.selectbox("Select the y-axis",options=columns+["None"],index=None)
        
        plot_list=["Line Plot","Bar Plot","Scatter Plot","Distribution Plot","Count Plot"]

        selected_plot=st.selectbox("select a Plot",options=plot_list,index=None)
        
        
    # button to generate plot
    if st.button("Generate Plot"):
        
        fig,ax=plt.subplots(figsize=(6,4))
        
        if selected_plot=="Line Plot":
            sns.lineplot(x=df[x_axis],y=df[y_axis],ax=ax)
            plt.title(label=f"{selected_plot} of {y_axis} vs {x_axis}",fontsize=12)
            plt.xlabel(x_axis,fontsize=10)
            plt.ylabel(y_axis,fontsize=10)
            
        elif selected_plot=="Bar Plot":
            sns.barplot(x=df[x_axis],ax=ax)
            plt.title(label=f"{selected_plot} of {y_axis} vs {x_axis}",fontsize=12)
            plt.xlabel(x_axis,fontsize=10)
            plt.ylabel(y_axis,fontsize=10)
        
        elif selected_plot=="Scatter Plot":
            sns.scatterplot(x=df[x_axis],y=df[y_axis],ax=ax)
            plt.title(label=f"{selected_plot} of {y_axis} vs {x_axis}",fontsize=12)
            plt.xlabel(x_axis,fontsize=10)
            plt.ylabel(y_axis,fontsize=10)
        
        elif selected_plot=="Distribution Plot":
            sns.distplot(x=df[x_axis],ax=ax)
            plt.title(label=f"{selected_plot} of {x_axis}",fontsize=12)
            plt.xlabel(x_axis,fontsize=10)
    
        
        elif selected_plot=="Count Plot":
            sns.countplot(x=df[x_axis],ax=ax)
            plt.title(label=f"{selected_plot} of {x_axis}",fontsize=12)
            plt.xlabel(x_axis,fontsize=10)
            
            
        #adjust the label sizes
        ax.tick_params(axis="x",labelsize=10)
        ax.tick_params(axis="y",labelsize=10)
        
        # title axis label
       # plt.title(label=f"{selected_plot} of {y_axis} vs {x_axis}",fontsize=12)
        #plt.xlabel(x_axis,fontsize=10)
        #plt.ylabel(y_axis,fontsize=10)
        
        st.pyplot(fig)
            
                









