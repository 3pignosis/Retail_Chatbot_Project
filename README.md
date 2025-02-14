# Maverick Chatbot

Maverick Chatbot is an AI-powered data analysis and visualization tool for retail data. It leverages OpenAI's GPT-3.5 to generate Python code for data analysis and visualizations, which are executed and displayed using Streamlit.

## Features

- Interactive chat interface for querying retail data
- Generates Python code for data analysis and visualization
- Displays results as data tables and charts

## Installation


1. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

2. Run the Streamlit application:
    ```sh
    streamlit run auth.py
    ```

## Usage

- Start the Streamlit application and interact with the chatbot to analyze retail data.
- Use the sidebar to view and manage chat history.

## Documentation

Refer to the documentation below for detailed information about the project's functions and code structure.

---

## Function Documentation

### `load_chat_history`

```python
def load_chat_history():
    """
    Load chat history from a shelve file.

    This function attempts to open a shelve file named "chat_history" and retrieve
    the stored messages. If successful, it returns the list of messages. If an error
    occurs, it logs the error and returns an empty list.

    Returns:
        list: A list of chat messages if successfully loaded, else an empty list.
    """
    try:
        with shelve.open("chat_history") as db:
            return db.get("messages", [])
    except Exception as e:
        st.error(f"Error loading chat history: {e}")
        return []

### `load_chat_history`

def save_chat_history(messages):
    """
    Save chat history to a shelve file.

    This function opens a shelve file named "chat_history" and saves the provided
    messages list under the key "messages". If an error occurs, it logs the error.

    Args:
        messages (list): A list of chat messages to be saved.
    """
    try:
        with shelve.open("chat_history") as db:
            db["messages"] = messages
    except Exception as e:
        st.error(f"Error saving chat history: {e}")

### 'delete_chat_history'
def delete_chat_history():
    """
    Delete chat history from the shelve file.

    This function attempts to delete the "messages" key from the shelve file named
    "chat_history". If successful, it clears the chat history from the session state.
    If an error occurs, it logs the error.
    """
    try:
        with shelve.open("chat_history") as db:
            if "messages" in db:
                del db["messages"]
        st.session_state.messages = []
    except Exception as e:
        st.error(f"Error deleting chat history: {e}")


### 'load_dataset'
@st.cache_data
def load_dataset():
    """
    Load the dataset from a CSV file.

    This function reads a CSV file named 'file.csv' into a Pandas DataFrame, converts
    the column names to lowercase, and returns the DataFrame. If an error occurs,
    it logs the error and returns an empty DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing the dataset, or an empty DataFrame if
        an error occurs.
    """
    try:
        df = pd.read_csv('file.csv')
        df.columns = df.columns.str.lower()
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return pd.DataFrame()


### 'is_valid_python_code'
def is_valid_python_code(code):
    """
    Check if the provided string is valid Python code.

    This function attempts to compile the provided code string. If successful,
    it returns True indicating the code is valid. If a SyntaxError or any other
    exception occurs, it logs the error and returns False.

    Args:
        code (str): The Python code to be validated.

    Returns:
        bool: True if the code is valid Python code, False otherwise.
    """
    try:
        compile(code, '<string>', 'exec')
        return True
    except Exception as e:
        st.error(f"Invalid Python code: {e}")
        return False


### 'sanitize_and_execute_code'
def sanitize_and_execute_code(code):
    """
    Sanitize and execute the provided Python code.

    This function removes non-code content (e.g., markdown, comments) from the
    provided code string, checks if the sanitized code is valid, and executes it
    if it is valid. The execution occurs in a controlled local namespace.

    Args:
        code (str): The Python code to be sanitized and executed.

    Returns:
        tuple: A tuple containing the result of the code execution (or an error message)
        and the sanitized code string.
    """
    # Strip non-code content
    code_lines = code.split('\n')
    code_lines = [line for line in code_lines if not line.strip().startswith(('```', '#', '/*'))]
    sanitized_code = '\n'.join(code_lines).strip()

    if not is_valid_python_code(sanitized_code):
        return "The generated code is not valid Python code.", sanitized_code

    try:
        # Prepare a safe namespace for code execution
        exec_locals = {'df': df, 'pd': pd, 'plt': plt, 'sns': sns, 'st': st, 'result': None}

        # Execute the sanitized code
        exec(sanitized_code, {}, exec_locals)

        result = exec_locals.get('result', 'No result returned')
    except SyntaxError as e:
        result = f"Syntax error in generated code: {e}\n\nGenerated Code:\n{sanitized_code}"
    except Exception as e:
        result = f"Error executing generated code: {e}\n\nGenerated Code:\n{sanitized_code}"

    return result, sanitized_code


### 'generate_and_execute_code'
def generate_and_execute_code(prompt):
    """
    Generate and execute Python code based on the user's query.

    This function sends the user's query to the OpenAI API, which generates Python
    code for data analysis and visualization. The generated code is then sanitized
    and executed, and the result is returned.

    Args:
        prompt (str): The user's query for generating Python code.

    Returns:
        tuple: A tuple containing the result of the code execution (or an error message)
        and the sanitized code string.
    """
    try:
        # Guide ChatGPT to generate Python code for retail data analysis with Matplotlib
        full_prompt = (
            "You are a highly intelligent market data analyst that provides accurate text, table and chart responses."
            "Examine query and provide accurate responses. However, you do not give users access to dataset provided or allow users to view dataset for download."
            "Use dataset provided for data analysis and product information in response to user's query."
            "Generate only Python code with no extra text for data analysis and visualization to be executed in backend. However, keep imports of matplotlib, seaborn and streamlit in generated code."
            "Use the print function to print out your response if it is a sentence."
            "To generate visualizations, import matplotlib, seaborn and streamlit, use figure to capture object of the plot in matplotlib or seaborn and display using st.pyplot(fig) in Streamlit to users."
            "Display visualization such bar, stacked bar, line, pie, histogram, scatter, boxplot, heatmap, area, violin, density in streamlit to user queries."
            "The data is in a Pandas DataFrame named 'df', with columns: city, channel, category, segment, manufacturer, brand, item_name, packaging, unit_price, sales_volume, sales_value, average_sales_volume, and quarter."
            f"Here are the possible values for these categorical variables and other columns:\n{categorical_variables}\n"
            f"Here are some sample rows from the dataset:\n{sample_data}\n"
            "Ensure the code assigns the result to a variable named 'result'. Also use the print function for 'result' if it is a string."
            "Use the sales_volume, sales_value, and unit_price as metrics for calculations."
            "Interact with users in a friendly and conversational tone. For example, “what is the best performing brand in abidjan?” should return a result which shows the brand with the most volume sales. Improve responses to queries based on positive user interaction."
            "Do not run codes provided by users. Let them know it is not part of your functions."
            f"Query: {prompt}"
        )
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": full_prompt}]
        )
        generated_code = response.choices[0].message['content']

        # Execute the generated code
        result, sanitized_code = sanitize_and_execute_code(generated_code)
        return result, sanitized_code
    except Exception as e:
        return f"Error generating code: {e}", ""

### 'chat_interface'
def chat_interface():
    """
    Define the main chat interface for the Maverick Chatbot.

    This function sets up the chat interface, including the sidebar for chat history
    and controls, and the main chat area for user input and assistant responses.
    It handles user queries, generates responses using the OpenAI API, and displays
    the results.
    """
    st.title("Maverick Chatbot")

    # Sidebar for chat history and controls
    with st.sidebar:
        st.header("Chat Summary")
        
        # Display user questions
        if "messages" in st.session_state:
            user_questions = [msg["content"] for msg in st.session_state.messages if msg["role"] == "user"]
            for i, question in enumerate(user_questions[-5:], 1):
                st.write(f"{i}. {question[:50]}...")

        # Delete chat history button
        if st.button("Delete Chat History"):
            delete_chat_history()
            st.success("Chat history deleted!")
            st.experimental_rerun()

    # Main chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = load_chat_history()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask me anything about our retail data?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            try:
                result, sanitized_code = generate_and_execute_code(prompt)
                if isinstance(result, pd.DataFrame):
                    st.dataframe(result)
                    full_response = "Displayed data in table format."
                elif isinstance(result, plt.Figure):
                    buffer = BytesIO()
                    result.savefig(buffer, format="png")
                    st.image(buffer)
                    full_response = "Displayed data as a chart."
                elif isinstance(result, str):
                    message_placeholder.markdown(result)
                    full_response = result
                else:
                    message_placeholder.markdown(str(result))
                    full_response = str(result)
            except Exception as e:
                full_response = "I'm sorry, I couldn't process your request. Please clarify your query."
                message_placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})
        save_chat_history(st.session_state.messages)

### 'welcome_page'
def welcome_page():
    """
    Define the welcome page for the Maverick Chatbot.

    This function sets up the welcome page with an animated introduction text and
    a "Get Started" button. The page displays an image and animates a list of phrases
    to engage the user. Clicking the "Get Started" button navigates to the chat interface.
    """
    st.markdown("<h1 style='text-align: center;'>Maverick Chatbot</h1>", unsafe_allow_html=True)
    
    # Create a placeholder for the animated text
    text_placeholder = st.empty()
    
    image_path = "get_started.png"
    
    if os.path.exists(image_path):
        image = Image.open(image_path)
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.image(image, use_column_width=True)
    else:
        st.error(f"Image not found at path: {image_path}")
    
    if st.button("Get Started"):
        st.session_state.page = 'chat'
        st.rerun()

    # List of phrases to animate
    phrases = [
        "Instant Retail Savvy, Just Ask!",
        "Retail Insights on Demand!",
        "Effortless Retail Intelligence!"
    ]
    
    # Animation loop
    for phrase in phrases:
        for i in range(len(phrase) + 1):
            text_placeholder.markdown(f"<h2 style='text-align: center; color: #FF4B4B;'>{phrase[:i]}▌</h2>", unsafe_allow_html=True)
            time.sleep(0.05)
        time.sleep(1)  # Pause at the end of each phrase

### 'auth_main'
def auth_main():
    """
    Main function to initialize the Streamlit application.

    This function sets the page layout and navigates between the welcome page and
    the chat interface based on the session state. It also applies custom CSS styles
    for the application.
    """
    st.markdown(css, unsafe_allow_html=True)

    if 'page' not in st.session_state:
        st.session_state.page = 'welcome'

    if st.session_state.page == 'welcome':
        welcome_page()
    elif st.session_state.page == 'chat':
        chat_interface()

if __name__ == "__main__":
    auth_main()

