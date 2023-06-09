# Landmark-Retrieval-AI
Our final project for the Information Retrieval course. We made a tool to retrieve similar images from a landmark database using feature matching.

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
  
Alternatively, you can just pull the data branch instead of the main branch.  

# How to use
To run a test query and generate a CSV file comprising of 300 queries, each outputing 10 top results, run `python model_lib.py`.  
To use the UI app, run `streamlit run app.py`.  

# Screenshot
![image](https://user-images.githubusercontent.com/55624202/235338857-c1d16dd3-12f8-4d8d-8bcd-5f44e283ef8d.png)
