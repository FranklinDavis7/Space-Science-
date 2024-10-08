import sklearn
import google.generativeai as genai
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt  # Correct import for submodule
from sklearn.preprocessing import OneHotEncoder
from streamlit_lottie import st_lottie
import requests
import time

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
lottie_coding=load_lottieurl("https://lottie.host/e2811820-050a-42ac-a4e6-a97aa1507767/tq5zHwv0BD.json")

# Replace with your actual API key
genai.configure(api_key="AIzaSyDT7o35Z51qAzMct1iLQaqf2AkVEC1AQaM")

st.sidebar.header("Filter (Filters the dataset values displayed)")

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
    dfselect = st.selectbox('Select a CSV file', ['Sample', 'Assay'])
    if dfselect == 'Sample':
        column_names = df1.columns.tolist()
        selected_columns = st.multiselect('Select Columns for Pie Chart', column_names)
        if st.button('Generate Plot'):
            if len(selected_columns) > 0:
                # Apply one-hot encoding to selected columns
                encoder = OneHotEncoder()  # No `sparse_out` argument
                encoded_data = encoder.fit_transform(df1[selected_columns]).toarray()  # Use `.toarray()` to convert to dense array
                encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(selected_columns))

                fig, ax = plt.subplots(figsize=(6, 8))  # Adjust figure size for portrait
                wedges, texts = ax.pie(encoded_df.sum(), labels=None, autopct=None, startangle=90)  # Remove autotexts
                ax.axis('equal')

                # Calculate percentages
                percentages = (encoded_df.sum() / encoded_df.sum().sum() * 100).round(1)

                # Create a legend with color patches and percentages
                legend_patches = [plt.matplotlib.patches.Patch(color=wedge.get_facecolor(), label=f"{label} ({percent}%)")
                                  for wedge, label, percent in zip(wedges, encoded_df.columns, percentages)]
                plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
                st.pyplot(fig)
            else:
                st.warning("Please select at least one column for the pie chart.")
    elif dfselect == 'Assay':
        column_names = df2.columns.tolist()
        selected_columns = st.multiselect('Select Columns for Pie Chart', column_names)
        if st.button('Generate Plot'):
            if len(selected_columns) > 0:
                # Apply one-hot encoding to selected columns
                encoder = OneHotEncoder()  # No `sparse_out` argument
                encoded_data = encoder.fit_transform(df2[selected_columns]).toarray()  # Use `.toarray()` to convert to dense array
                encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(selected_columns))

                fig, ax = plt.subplots(figsize=(6, 8))  # Adjust figure size for portrait
                wedges, texts = ax.pie(encoded_df.sum(), labels=None, autopct=None, startangle=90)  # Remove autotexts
                ax.axis('equal')

                # Calculate percentages
                percentages = (encoded_df.sum() / encoded_df.sum().sum() * 100).round(1)

                # Create a legend with color patches and percentages
                legend_patches = [plt.matplotlib.patches.Patch(color=wedge.get_facecolor(), label=f"{label} ({percent}%)")
                                  for wedge, label, percent in zip(wedges, encoded_df.columns, percentages)]
                plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
                st.pyplot(fig)
            else:
                st.warning("Please select at least one column for the pie chart.")
def scatter():
    # Get the DataFrames from session state
    df1 = st.session_state.df1
    df2 = st.session_state.df2

    # Get the column names from the combined DataFrame
    dfselect = st.selectbox('Select a CSV file', ['Sample', 'Assay'])
    if dfselect == 'Sample':
        column_names = df1.columns.tolist()
        selected_column1 = st.selectbox('Select X-Axis', column_names)
        selected_column2 = st.selectbox('Select Y-Axis', column_names)
        if st.button('Generate Plot'):
            # Create a scatter plot using matplotlib
            fig, ax = plt.subplots()

            # Sort the DataFrame based on the X-Axis column
            df1_sorted = df1.sort_values(by=selected_column1)

            # Plot the scatter plot using the sorted DataFrame
            ax.scatter(df1_sorted[selected_column1], df1_sorted[selected_column2], s=50, c='blue', alpha=0.7)
            ax.set_title('Scatter Plot')
            ax.set_xlabel(selected_column1)
            ax.set_ylabel(selected_column2)
            plt.xticks(rotation=45)
            st.pyplot(fig, use_container_width=True)

    elif dfselect == 'Assay':
        column_names = df2.columns.tolist()
        selected_column1 = st.selectbox('Select X-Axis', column_names)
        selected_column2 = st.selectbox('Select Y-Axis', column_names)
        if st.button('Generate Plot'):
            # Create a scatter plot using matplotlib
            fig, ax = plt.subplots()

            # Sort the DataFrame based on the X-Axis column
            df2_sorted = df2.sort_values(by=selected_column1)

            # Plot the scatter plot using the sorted DataFrame
            ax.scatter(df2_sorted[selected_column1], df2_sorted[selected_column2], s=50, c='red', alpha=0.7)
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
        selected_columns = st.multiselect('Select Columns for Histogram', column_names)  # Use the multiselect method
    elif dfselect == 'Assay':
        column_names = df2.columns.tolist()
        selected_columns = st.multiselect('Select Columns for Histogram', column_names) # Use the multiselect method

    if st.button('Generate Plot'):
        if len(selected_columns) > 0:
            # Use cached one-hot encoding
            encoded_df = encode_data(df1 if dfselect == 'Sample' else df2, selected_columns)

            fig, ax = plt.subplots(figsize=(10, 6))  # Increased figure size

            # Plot each histogram with a unique color
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'cyan']  # Add more colors as needed
            for i, column in enumerate(encoded_df.columns):
                ax.hist(encoded_df[column], label=column, alpha=0.5, width=0.5, color=colors[i % len(colors)])

            ax.set_title('Histogram')
            ax.set_xlabel('Values')
            ax.set_ylabel('Frequency')

            # Calculate percentages for each column
            percentages = (encoded_df.sum() / encoded_df.sum().sum() * 100).round(1)

            # Create a legend with color patches and percentages
            legend_patches = [plt.matplotlib.patches.Patch(color=color, label=f"{column} ({percent}%)")
                              for i, (column, color, percent) in enumerate(zip(encoded_df.columns, colors, percentages))]
            plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)  # Adjust legend position

            plt.xticks(rotation=45)
            plt.tight_layout()  # Adjusts spacing
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

        # Initialize ai_response1 and ai_response2 within this block
        # Generate AI response only once when files are uploaded
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

        st.session_state.ai_response1 = chat_session.send_message("provide the column name combination for visualization")
        st.session_state.ai_response2 = chat_session.send_message("a brief description about this experiment")


    st.title("Space Science Visualization")

    st.header("Suggestions ")
    st.write(st.session_state.ai_response2.text) 
    st.write(st.session_state.ai_response1.text)  # Print the Gemini response to the Streamlit page
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
    df2 = pd.read_csv("input_csv2.csv")

    # Filter Dataframes
    st.subheader("SAMPLE")
    filtered_df1 = df1.copy()
    column_names1 = df1.columns.tolist()
    for i, col in enumerate(column_names1):
        unique_values = df1[col].unique().tolist()
        selected_values = st.sidebar.multiselect(f"Select {col}:", options=unique_values, default=unique_values, key=f"multiselect_{col}_{i}")  # Add unique key
        if selected_values:
            filtered_df1 = filtered_df1[filtered_df1[col].isin(selected_values)]
    st.dataframe(filtered_df1)

    st.subheader("ASSAYS")
    filtered_df2 = df2.copy()
    column_names2 = df2.columns.tolist()
    for i, col in enumerate(column_names2):
        unique_values = df2[col].unique().tolist()
        selected_values = st.sidebar.multiselect(f"Select {col}:", options=unique_values, default=unique_values, key=f"multiselect_{col}_{i}")  # Add unique key
        if selected_values:
            filtered_df2 = filtered_df2[filtered_df2[col].isin(selected_values)]
    st.dataframe(filtered_df2)

    # Create a Pygwalker instance

if __name__ == "__main__":
    main()
