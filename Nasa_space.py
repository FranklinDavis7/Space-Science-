
import time
import google.generativeai as genai
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt  # Correct import for submodule
from sklearn.preprocessing import OneHotEncoder
from streamlit_lottie import st_lottie
import requests

def load_lottieurl(url):
    r = requests. get (url)
    if r.status_code != 200:
        return None
    return r.json()
lottie_coding=load_lottieurl("https://lottie.host/e2811820-050a-42ac-a4e6-a97aa1507767/tq5zHwv0BD.json")

# Replace with your actual API key
genai.configure(api_key="AIzaSyDT7o35Z51qAzMct1iLQaqf2AkVEC1AQaM")

st.sidebar.header("Filter (Use Generate Button to activate filter)")

def main():
    if "page" not in st.session_state:
        st.session_state.page = "home"
    if st.session_state.page == "home":
        home()
    elif st.session_state.page == "mainpage":
        mainpage()

def line():
    # Get the DataFrames from session state
    df1 = st.session_state.df1
    df2 = st.session_state.df2

    # Get the column names from the combined DataFrame
    dfselect=st.selectbox('Select a CSV file',['Sample','Assay'])
    if dfselect=='Sample':
        column_names=df1.columns.tolist()
        selected_column1 = st.selectbox('Select a X-Labe', column_names)
        selected_column2 = st.selectbox('Select a Y-Label', column_names)
        if st.button('Generate Plot'):
            # Create a line chart using matplotlib
            fig, ax = plt.subplots() 
            ax.plot(df1[selected_column1], df1[selected_column2])
            ax.set_title('Line Chart')
            ax.set_xlabel(selected_column1)
            ax.set_ylabel(selected_column2)
            plt.xticks(rotation=45) 
            st.pyplot(fig, use_container_width=True)

    elif dfselect=='Assay':
        column_names=df2.columns.tolist()
        selected_column1 = st.selectbox('Select a X-Labe', column_names)
        selected_column2 = st.selectbox('Select a Y-Label', column_names)
        if st.button('Generate Plot'):
            # Create a line chart using matplotlib
            fig, ax = plt.subplots()
            ax.plot(df2[selected_column1], df2[selected_column2])
            ax.set_title('Line Chart')
            ax.set_xlabel(selected_column1)
            ax.set_ylabel(selected_column2)
            plt.xticks(rotation=45) 
            st.pyplot(fig, use_container_width=True)


def bar():
    # Get the DataFrames from session state
    df1 = st.session_state.df1
    df2 = st.session_state.df2

    # Get the column names from the combined DataFrame
    dfselect=st.selectbox('Select a CSV file',['Sample','Assay'])
    if dfselect=='Sample':
        column_names=df1.columns.tolist()
        selected_column1 = st.selectbox('Select a X-Labe', column_names)
        selected_column2 = st.selectbox('Select a Y-Label', column_names)
        if st.button('Generate Plot'):
            # Create a bar chart using matplotlib
            fig, ax = plt.subplots() 
            ax.bar(df1[selected_column1], df1[selected_column2])
            ax.set_title('Bar Chart')
            ax.set_xlabel(selected_column1)
            ax.set_ylabel(selected_column2)
            plt.xticks(rotation=45) 
            st.pyplot(fig, use_container_width=True)

    elif dfselect=='Assay':
        column_names=df2.columns.tolist()
        selected_column1 = st.selectbox('Select a X-Labe', column_names)
        selected_column2 = st.selectbox('Select a Y-Label', column_names)
        if st.button('Generate Plot'):
            # Create a bar chart using matplotlib
            fig, ax = plt.subplots()
            ax.bar(df2[selected_column1], df2[selected_column2])
            ax.set_title('Bar Chart')
            ax.set_xlabel(selected_column1)
            ax.set_ylabel(selected_column2)
            plt.xticks(rotation=45) 
            st.pyplot(fig, use_container_width=True)

def pie():
    # Get the DataFrames from session state
    df1 = st.session_state.df1
    df2 = st.session_state.df2

    # Get the column names from the combined DataFrame
    dfselect=st.selectbox('Select a CSV file',['Sample','Assay'])
    if dfselect=='Sample':
        column_names=df1.columns.tolist()
        selected_columns = st.multiselect('Select Columns for Pie Chart', column_names)
        if st.button('Generate Plot'):
            if len(selected_columns) > 0:
                # Apply one-hot encoding to selected columns
                encoder = OneHotEncoder()  # No `sparse_out` argument
                encoded_data = encoder.fit_transform(df1[selected_columns]).toarray()  # Use `.toarray()` to convert to dense array
                encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(selected_columns))

                fig, ax = plt.subplots()
                ax.pie(encoded_df.sum(), labels=encoded_df.columns, autopct='%1.1f%%', startangle=90)
                ax.axis('equal')
                st.pyplot(fig)
            else:
                st.warning("Please select at least one column for the pie chart.")
    elif dfselect=='Assay':
        column_names=df2.columns.tolist()
        selected_columns = st.multiselect('Select Columns for Pie Chart', column_names)
        if st.button('Generate Plot'):
            if len(selected_columns) > 0:
                # Apply one-hot encoding to selected columns
                encoder = OneHotEncoder()  # No `sparse_out` argument
                encoded_data = encoder.fit_transform(df2[selected_columns]).toarray()  # Use `.toarray()` to convert to dense array
                encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(selected_columns))

                fig, ax = plt.subplots()
                ax.pie(encoded_df.sum(), labels=encoded_df.columns, autopct='%1.1f%%', startangle=90)
                ax.axis('equal')
                st.pyplot(fig)
            else:
                st.warning("Please select at least one column for the pie chart.")

def scatter():
    # Get the DataFrames from session state
    df1 = st.session_state.df1
    df2 = st.session_state.df2

    # Get the column names from the combined DataFrame
    dfselect=st.selectbox('Select a CSV file',['Sample','Assay'])
    if dfselect=='Sample':
        column_names=df1.columns.tolist()
        selected_column1 = st.selectbox('Select a X-Labe', column_names)
        selected_column2 = st.selectbox('Select a Y-Label', column_names)
        if st.button('Generate Plot'):
            # Create a scatter plot using matplotlib
            fig, ax = plt.subplots() 
            ax.scatter(df1[selected_column1], df1[selected_column2])
            ax.set_title('Scatter Plot')
            ax.set_xlabel(selected_column1)
            ax.set_ylabel(selected_column2)
            plt.xticks(rotation=45) 
            st.pyplot(fig, use_container_width=True)

    elif dfselect=='Assay':
        column_names=df2.columns.tolist()
        selected_column1 = st.selectbox('Select a X-Labe', column_names)
        selected_column2 = st.selectbox('Select a Y-Label', column_names)
        if st.button('Generate Plot'):
            # Create a scatter plot using matplotlib
            fig, ax = plt.subplots()
            ax.scatter(df2[selected_column1], df2[selected_column2])
            ax.set_title('Scatter Plot')
            ax.set_xlabel(selected_column1)
            ax.set_ylabel(selected_column2)
            plt.xticks(rotation=45) 
            st.pyplot(fig, use_container_width=True)

@st.cache_data
def encode_data(df, selected_columns):
    """One-hot encodes the selected columns and caches the result."""
    encoder = OneHotEncoder()
    encoded_data = encoder.fit_transform(df[selected_columns]).toarray()
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(selected_columns))
    return encoded_df

def histogram():
    # Get the DataFrames from session state
    df1 = st.session_state.df1
    df2 = st.session_state.df2

    # Get the column names from the combined DataFrame
    dfselect = st.selectbox('Select a CSV file', ['Sample', 'Assay'])
    
    # Display sidebar multiselect only if dfselect is chosen
    if dfselect == 'Sample':
        column_names = df1.columns.tolist()
        selected_columns = st.sidebar.multiselect("Select column name:", options=column_names, default=column_names)
    elif dfselect == 'Assay':
        column_names = df2.columns.tolist()
        selected_columns = st.sidebar.multiselect("Select column name:", options=column_names, default=column_names)

    if st.button('Generate Plot'):
        if len(selected_columns) > 0:
            # Use cached one-hot encoding
            encoded_df = encode_data(df1 if dfselect == 'Sample' else df2, selected_columns)

            fig, ax = plt.subplots(figsize=(10, 6))
            for column in encoded_df.columns:
                ax.hist(encoded_df[column], label=column, alpha=0.5, width=0.5)
            ax.set_title('Histogram')
            ax.set_xlabel('Values')
            ax.set_ylabel('Frequency')
            ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("Please select at least one column for the histogram.")

def home():

    st.title("BioVis ")
    left_column,rightcolumn=st.columns(2)

    # Set space blue background color
    st.markdown(
        """
        <style>
        body {
            background-color: #002040;  /* Space blue color */
            color: #fff;   /* White text color */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Embed the Lottie animation
    with rightcolumn:
        st.lottie(lottie_coding,height=500 ,width=400,key="coding")

    st.session_state.input_csv1 = st.file_uploader("Upload The CSV File Sample", type=['csv'])
    st.session_state.input_csv2 = st.file_uploader("Upload the CSV File Assay", type=['csv'])
    if st.session_state.input_csv1 is not None and st.session_state.input_csv2 is not None:
        if st.button("Submit"):
            st.session_state.page = "mainpage"
            # Load DataFrames into session state
            st.session_state.df1 = pd.read_csv(st.session_state.input_csv1) 
            st.session_state.df2 = pd.read_csv(st.session_state.input_csv2)

def mainpage():
    if "uploaded_files" not in st.session_state:
        def upload_to_gemini(path, mime_type=None):
            """Uploads the given file to Gemini."""
            file = genai.upload_file(path, mime_type=mime_type)
            print(f"Uploaded file '{file.display_name}' as: {file.uri}")
            return file

        def wait_for_files_active(files):
            """Waits for the given files to be active."""
            print("Waiting for file processing...")
            for name in (file.name for file in files):
                file = genai.get_file(name)
                while file.state.name == "PROCESSING":
                    print(".", end="", flush=True)
                    time.sleep(10)
                    file = genai.get_file(name)
                if file.state.name != "ACTIVE":
                    raise Exception(f"File {file.name} failed to process")
            print("...all files ready")
            print()

        # Create the model
        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }

        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config,
            system_instruction="Generate only the most relevant answer.Give the answers in one line without much description",
        )

        # Save the files to the local file system
        with open("input_csv1.csv", "wb") as f:
            f.write(st.session_state.input_csv1.getvalue())
        with open("input_csv2.csv", "wb") as f:
            f.write(st.session_state.input_csv2.getvalue())

        # TODO Make these files available on the local file system
        # You may need to update the file paths
        files = [
            upload_to_gemini("input_csv1.csv", mime_type="text/csv"),
            upload_to_gemini("input_csv2.csv", mime_type="text/csv"),
        ]

        # Some files have a processing delay. Wait for them to be ready.
        wait_for_files_active(files)

        st.session_state.uploaded_files = files

    chat_session = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config={
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        },
        system_instruction="Generate only the most relevant answer.Give the answers in one line without much description",
    ).start_chat(
        history=[
            {
                "role": "user",
                "parts": [
                    st.session_state.uploaded_files[0],
                    st.session_state.uploaded_files[1],
                    "provide the column name combination and idea for visualization like bar graphs,heat maps, line charts etc ",
                    "rerun\n",
                ],
            },
        ]
    )

    response = chat_session.send_message("provide the column name combination for visualization")
    response2=chat_session.send_message("a brief description about this experiment")
    st.title("Space Science Visualization")

    st.header("Description ")
    st.write(response2.text)
    st.subheader("Suggestions")
    st.write(response.text)  # Print the Gemini response to the Streamlit page
    # Create a list of options


    # Create a list of options
    options = ['Line Graph', 'Bar Graph', 'Pie Chart','Scatter Plot','Histogram']

    # Create a selectbox
    selected_option = st.selectbox('Select an option', options)
    if selected_option =='Line Graph':
        line()
    elif selected_option =='Bar Graph':
        bar()
    elif selected_option =='Pie Chart':
        pie()
    elif selected_option =='Scatter Plot':
        scatter()
    elif selected_option =='Histogram':
        histogram()
    
    # Display the selected option
    st.write(f'Selected option: {selected_option}')

    df1 = pd.read_csv("input_csv1.csv")

    df2=pd.read_csv("input_csv2.csv")
    # ... (rest of your Streamlit code)
    st.subheader("SAMPLE")
    st.dataframe(df1)
    st.subheader("ASSAYS")
    st.dataframe(df2)
    # Create a Pygwalker instance

if __name__ == "__main__":
    main()
