# Landmark-Retrieval-AI
Our final project for the Information Retrieval course. We made a tool to retrieval similar images from a database using an feature matching.

# Introduction
The original model (**original_trained_model.ipynb**) was made on Google Colab. A version of it (**model_lib.py**) was revised and edited to make an interface for the app.  
The UI was made using Streamlit.  

# Data
Before running the app, download the data from here: https://drive.google.com/file/d/1nQIfFMeQuq2rmYHcYAZGMqpROP4s8kAO/view?usp=sharing  
Copy the folders **database** and **query** from **Landmark_Retrieval/test/** to **./data/**.  
  
The required folder structure is:  
  
data/  
--checkpoint-500   
--database/  
--query  

# How to use
To run a test query and generate a CSV file comprising of 300 queries, each outputing 10 top results, run **python model_lib.py**.  
To use the UI app, run **streamlit run app.py**.  
