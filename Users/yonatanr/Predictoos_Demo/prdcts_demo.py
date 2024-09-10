import streamlit as st
import pandas as pd
import time 
from openai import AzureOpenAI
import os
from datetime import date,timedelta,datetime


# Display the logo
st.markdown("![](https://www.diplomat-global.com/wp-content/uploads/2018/06/logo.png)")

# Set the title of the app
st.title('Predictoos AI Hub')

def df_explainer(df):  
    """  
    Takes a pandas DataFrame and returns a summary of its contents.  
      
    Parameters:  
    - df: pd.DataFrame, the DataFrame to summarize.  
  
    Returns:  
    - summary: str, a summary of the DataFrame contents.  
    """  
    try:  
        # Check if the input is a DataFrame  
        if not isinstance(df, pd.DataFrame):  
            raise ValueError("Input must be a pandas DataFrame.")  
          
        # Get basic information about the DataFrame  
        summary = []  
        summary.append(f"Number of Rows: {len(df)}")  
        summary.append(f"Number of Columns: {df.shape[1]}")  
        summary.append(f"Columns: {', '.join(df.columns)}")  
          
        # Add data types of each column  
        summary.append("Data Types:")  
        for col, dtype in df.dtypes.items():  
            summary.append(f"  - {col}: {dtype}")  
          
        # Add basic statistics for all columns  
        summary.append("Summary Statistics for All Columns:")  
        summary.append(df.describe(include='all').to_string())  
          
        # Generate 5 random sorted samples  
        random_samples = df.sample(n=5, random_state=1).sort_index()  # Using random_state for reproducibility  
        summary.append("Random Samples:")  
        for index, row in random_samples.iterrows():  
            for i,col in enumerate(df.columns):
                if i!=0:
                    summary.append(f"  - [{col}] - {row[col]}")  
                else:
                    summary.append(f"|sample - [{col}] - {row[col]}")
          
        return "\n".join(summary)  
      
    except Exception as e:  
        return f"An error occurred: {e}" 


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


# File uploader for CSV files
# A visually attractive note using Markdown for instructions
st.markdown(
    """
    <div style="background-color: #f9f9f9; padding: 10px; border-radius: 5px; border: 1px solid #e0e0e0;">
        <h3 style="color: #1f77b4;">📊 Upload Your Sales Data</h3>
        <p style="font-size: 18px;">Please upload a single CSV file containing:</p>
        <ul style="font-size: 16px;">
            <li><strong>Date:</strong> The date of the sale</li>
            <li><strong>Material Number:</strong> The identifier for the item sold</li>
            <li><strong>Sale Quantity:</strong> The number of items sold</li>
        </ul>
        <p style="color: #ff7f0e;">&nbsp;&nbsp;&nbsp; File format: <strong>CSV</strong></p>
        <p style="color: #ff7f0e;">&nbsp;&nbsp;&nbsp; Max file size: <strong>10 MB</strong></p>
    </div>
    """,
    unsafe_allow_html=True
)

# File uploader with a hidden label
uploaded_file = st.file_uploader("", type="csv", label_visibility="collapsed")


# upload_file_txt = "For the demo process please upload a CSV file"
# uploaded_file = st.file_uploader(upload_file_txt, type=["csv"], label_visibility="hidden")

if uploaded_file is not None:

    # Read the CSV file into a DataFrame
    df = pd.read_csv(uploaded_file)

    # Display the first 5 rows of the DataFrame
    st.write("Here are the first 5 rows of the uploaded CSV:")
    st.dataframe(df.head())

    # Suggest "date","store" and "barcode" and "sales" columns
    df_explainer_txt = df_explainer(df)
    # columns_to_find = ['date', 'store', 'barcode', 'sales quantity'] 
    columns_to_find = ['date', 'barcode', 'sales quantity'] 
    column_explained = [  
        'main date of the file',  
        # 'the store id can be represented as customer id or code, and other terminology directed to a sub-chain level',  
        'First, try to find the most likely item or product name. If that isn’t available, then attempt to find the most likely product ID, which can be associated with the barcode, material ID, or code.',  
        'the sales quantity can be units, cartons, etc.'  
    ]  
    
    finder_texts = {}  
    
    for column, explanation in zip(columns_to_find, column_explained):  
        finder_sys = (f"You are an AI that has the sole purpose of finding the main {column} column in the pandas dataframe presented to you. "  
                    f"You answer only in the name of the column that represents it. "  
                    f"Note: {explanation}")  
    
        finder_texts[column] = generate_text(df_explainer_txt, finder_sys)  

    
    date_candidate = next((col for col in df.columns if finder_texts["date"] in col), None)
    # store_candidate = next((col for col in df.columns if finder_texts["store"] in col), None)
    barcode_candidate = next((col for col in df.columns if finder_texts["barcode"] in col), None)
    sales_candidate = next((col for col in df.columns if finder_texts["sales quantity"] in col), None)

    # Radio button for choosing forecast type
    forecast_type = st.radio("Select Forecast Type:", ("Weekly", "Monthly"))


    # Allow the user to select the date column
    date_column = st.selectbox("Select the date column", options=df.columns.tolist(), index=df.columns.tolist().index(date_candidate) if date_candidate else 0)

    # Allow the user to select the store column
    # store_column = st.selectbox("Select the store column", options=df.columns.tolist(), index=df.columns.tolist().index(store_candidate) if store_candidate else 0)

    # Allow the user to select the barocde column
    barcode_column = st.selectbox("Select the barcode column", options=df.columns.tolist(), index=df.columns.tolist().index(barcode_candidate) if barcode_candidate else 0)

    # Allow the user to select the sales column with a default suggestion
    sales_column = st.selectbox("Select the sales column", options=df.columns.tolist(), index=df.columns.tolist().index(sales_candidate) if sales_candidate else 0)

    # Display the selected horizon for confirmation
    st.write(f"Selected horizon: {forecast_type}")
    
    # Display the selected columns for confirmation
    st.write(f"Selected date column: {date_column}")
    # st.write(f"Selected store column: {store_column}")
    st.write(f"Selected barcode column: {barcode_column}")
    st.write(f"Selected sales column: {sales_column}")


    

    # Save the confirmed selections
    if st.button("Confirm selections"):
        # Add a loading spinner with a delay
        with st.spinner('Calculating predictions...'):
            time.sleep(5)  # Wait for 5 seconds
            
        st.session_state.selected_date_column = date_column
        # st.session_state.selected_store_column = store_column
        st.session_state.selected_barcode_column = barcode_column
        st.session_state.selected_sales_column = sales_column

        st.session_state.forecast_type = forecast_type

        st.session_state.finder_texts = {'date':st.session_state.selected_date_column,
                                        #  'store':st.session_state.selected_store_column,
                                         'barcode':st.session_state.selected_barcode_column, 
                                         'sales quantity':st.session_state.selected_sales_column}

        st.success("Selections saved!")

        df = df[list(st.session_state.finder_texts.values())]
        # begin converting the data to fit the model
        
        # date conversion
        df[st.session_state.selected_date_column] = pd.to_datetime(df[st.session_state.selected_date_column])
        
        #max date of the data
        max_dt = df[st.session_state.selected_date_column].max()

        horizon = 7 if st.session_state.forecast_type=='Weekly' else 30

        history = 180 if st.session_state.forecast_type=='Weekly' else 360
        
        # taking leading barcodes
        grp_df = df.groupby(st.session_state.selected_barcode_column).agg({st.session_state.selected_sales_column:'sum',st.session_state.selected_date_column:lambda x: (max_dt - max(x)).days}).sort_values('Sales_Qty',ascending = False)

        # only sold in the period of the horizon
        barcode_lst = grp_df[grp_df[st.session_state.selected_date_column]<horizon].head(50).index.tolist()

        history_cutoff = max_dt - timedelta(days = history)

        # filter to the barcodes
        df = df[(df[st.session_state.selected_barcode_column].isin(barcode_lst))&(df[st.session_state.selected_date_column]>history_cutoff)]

        # fill in 0's of non sale days
        dt_df=df[[st.session_state.selected_barcode_column]].drop_duplicates(subset = [st.session_state.selected_barcode_column])
        full_dt_range = pd.date_range(df[st.session_state.selected_date_column].min(),df[st.session_state.selected_date_column].max())
        dt_df[st.session_state.selected_date_column] = [full_dt_range]*len(dt_df)


        # add the 0's automatically
        df = dt_df.explode(st.session_state.selected_date_column).merge(df,how = 'left',on = dt_df.columns.tolist()).fillna(0)


    # Display the selected columns alongside the original
    if 'finder_texts' in st.session_state:
        selected_df = df[df[st.session_state.selected_date_column]==df[st.session_state.selected_date_column].max()][list(st.session_state.finder_texts.values())]
        # selected_df = df.head(10)
        st.write("Here is a sample of your demand forecast:")
        st.dataframe(selected_df.head(10))

        # Function to convert DataFrame to CSV in memory
        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode('utf-8')

        # Create a download button for the selected DataFrame
        csv = convert_df_to_csv(selected_df)
        st.download_button(
            label="Download full forecast",
            data=csv,
            file_name='forecast.csv',
            mime='text/csv'
        )