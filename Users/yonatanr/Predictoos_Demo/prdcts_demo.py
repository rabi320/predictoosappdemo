import streamlit as st
import pandas as pd
import time 
from openai import AzureOpenAI
import os
from datetime import date,timedelta,datetime
import json
import time
from datetime import datetime,date,timedelta
import urllib


# Display the image from the URL
st.markdown("![](https://i0.wp.com/predictoos.com/wp-content/uploads/2024/07/LOGO-2-min.png?fit=159%2C33&ssl=1)")

# Set the title of the app
st.title("AI Inventory Strategy")


openai_api_key = os.getenv('OPENAI_KEY')

client = AzureOpenAI(  
    azure_endpoint="https://ai-usa.openai.azure.com/",  
    api_key=openai_api_key,  
    api_version="2024-02-15-preview"  
)  
MODEL = "Diplochat"  

  
def generate_text(prompt, sys_msg, examples=[]):  
    response = client.chat.completions.create(  
        model=MODEL,  # model = "deployment_name"  
        messages=[{"role": "system", "content": sys_msg}] + examples + [{"role": "user", "content": prompt}],  
        temperature=0.7,  
        max_tokens=2000,  
        top_p=0.95,  
        frequency_penalty=0,  
        presence_penalty=0,  
        stop=None  
    )  
    return response.choices[0].message.content.strip()  



inv_st_df = pd.read_csv('Users/yonatanr/Predictoos_Demo/Inventory_strategy_table.csv')

unique_skus = ','.join(inv_st_df.MATERIAL_NUMBER.unique().tolist())
st.text(unique_skus)

# Display the DataFrame with text wrapping
st.dataframe(inv_st_df, use_container_width=True)