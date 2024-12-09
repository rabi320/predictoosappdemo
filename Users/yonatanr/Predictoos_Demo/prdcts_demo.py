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

# unique_skus = ','.join(inv_st_df.MATERIAL_NUMBER.unique().tolist())
# st.text(unique_skus)

# Display the DataFrame with text wrapping
# st.dataframe(inv_st_df, use_container_width=True)

# Title of the app
st.title("Filter DataFrame by Unique Combinations of Material and Customer")

# Create a list of unique combinations
unique_combinations = inv_st_df[['MATERIAL_NAME', 'CUSTOMER_CODE']].drop_duplicates()

# Create checkboxes for each unique combination
selected_combinations = []

st.subheader("Select Unique Combinations (material_name, customer_code)")

for index, row in unique_combinations.iterrows():
    combination = f"{row['MATERIAL_NAME']} ({row['CUSTOMER_CODE']})"
    if st.checkbox(combination):
        selected_combinations.append((row['MATERIAL_NAME'], row['CUSTOMER_CODE']))

# Filter DataFrame based on selected combinations
if selected_combinations:
    filtered_df = inv_st_df[inv_st_df.apply(lambda x: (x['MATERIAL_NAME'], x['CUSTOMER_CODE']) in selected_combinations, axis=1)]
    st.write("Filtered DataFrame:")
    st.write(filtered_df)
else:
    st.write("Please select at least one combination to display the data.")


# Sample DataFrame
data = {
    'Material Name': ['Material A', 'Material B', 'Material C'],
    'Customer Code': ['C001', 'C002', 'C003']
}

df = pd.DataFrame(data)

# Title of the app
st.title("DataFrame with Tooltips")

# Display the DataFrame with tooltips
st.markdown("""
<style>
.tooltip {
    position: relative;
    display: inline-block;
    cursor: pointer;
}

.tooltip .tooltiptext {
    visibility: hidden;
    width: 120px;
    background-color: black;
    color: #fff;
    text-align: center;
    border-radius: 5px;
    padding: 5px;
    position: absolute;
    z-index: 1;
}

.tooltip:hover .tooltiptext {
    visibility: visible;
}
</style>
""", unsafe_allow_html=True)

# Display DataFrame with tooltips
for index, row in df.iterrows():
    tooltip_text = f"This is {row['Material Name']}, identified by code {row['Customer Code']}"
    st.write(f"<div class='tooltip'>{row['Material Name']}<span class='tooltiptext'>{tooltip_text}</span></div>",
             unsafe_allow_html=True)