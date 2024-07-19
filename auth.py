import streamlit as st
import shelve
import openai
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from PIL import Image
import os
import time

# Set page config as the first Streamlit command
st.set_page_config(page_title="Maverick Chatbot")

openai.api_key = st.secrets["OPENAI_API_KEY"]

# Load chat history from shelve file
def load_chat_history():
    try:
        with shelve.open("chat_history") as db:
            return db.get("messages", [])
    except Exception as e:
        st.error(f"Error loading chat history: {e}")
        return []

# Save chat history to shelve file
def save_chat_history(messages):
    try:
        with shelve.open("chat_history") as db:
            db["messages"] = messages
    except Exception as e:
        st.error(f"Error saving chat history: {e}")

# Delete chat history
def delete_chat_history():
    try:
        with shelve.open("chat_history") as db:
            if "messages" in db:
                del db["messages"]
        st.session_state.messages = []
    except Exception as e:
        st.error(f"Error deleting chat history: {e}")

# Load the dataset
@st.cache_data
def load_dataset():
    try:
        df = pd.read_csv('file.csv')
        df.columns = df.columns.str.lower()
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return pd.DataFrame()

df = load_dataset()
sample_data = df.head().to_dict(orient='records') if not df.empty else []

# Detailed information about the categorical variables and other columns
categorical_variables = {
    "city": ["Abidjan", "Bouake"],
    "channel": ["Boutique", "Groceries", "Open_Market"],
    "category": ["PASTA"],
    "segment": ["DRY PASTA"],
    "manufacturer": ["CAPRA", "GOYMEN FOODS", "DOUBA", "PAGANINI", "PANZANI", "PASTA DOUBA", "MR COOK", "TAT MAKARNACILIK SANAYI VE TICARET AS", "REINE", "MOULIN MODERNE", "AVOS GROUP", "OBA MAKARNA"],
    "brand": ["ALYSSA", "MAMAN", "BLE D'OR", "MONDO", "DOUBA", "PAGANINI", "PANZANI", "PASTA DOUBA", "PASTA AROMA", "BONJOURNE", "TAT MAKARNA", "PASTA MONDO", "REINE", "PASTA BOUBA", "GOUSTA", "OBA MAKARNA"],
    "item_name": [
        "ALYSSA SPAGHETTI 200G SACHET", "MAMAN SUPERIOR QUALITY FOOD PASTA 200G SACHET",
        "MAMAN VERMICELLI 200G SACHET", "MAMAN 1.1 SPAGHETTI 200G SACHET",
        "MAMAN 1.5 SPAGHETTI 200G SACHET", "BLE D'OR 200G SACHET",
        "MAMAN SPAGHETTI 200G SACHET", "MAMAN 1.5 SPAGHETTI 500G SACHET",
        "MONDO SPAGHETTI 500G SACHET", "MAMAN SPAGHETTI 4540G BAG",
        "MAMAN COQUILLETTES 200G SACHET", "DOUBA 500G SACHET",
        "PAGANINI SPAGHETTI 200G SACHET", "PANZANI CAPELLINI 500G SACHET",
        "PASTA DOUBA SPAGHETTI 500G SACHET", "BLE D'OR SPAGHETTI 200G SACHET",
        "PASTA AROMA 200G SACHET", "MAMAN COQUILLETTES 4540G BAG",
        "MAMAN VERMICELLI SUPERIOR QUALITY FOOD PASTA 4540G BAG", "MAMAN SPAGHETTI 500G SACHET",
        "MAMAN VERMICELLI 500G SACHET", "BONJOURNE SPAGHETTI 500G SACHET",
        "MAMAN SPAGHETTI 475G SACHET", "PANZANI GOLD SPAGHETTI QUALITY 250G SACHET",
        "MAMAN MACARONI 200G SACHET", "MAMAN SPAGHETTI 450G SACHET",
        "TAT MAKARNA SPAGHETTI 500G SACHET", "PASTA MONDO SPAGHETTI 200G SACHET",
        "REINE PASTA 500G SACHET", "PASTA BOUBA 500G SACHET",
        "BONJOURNE SPAGHETTI 200G SACHET", "MAMAN 200G SACHET",
        "GOUSTA SPAGHETTI ALTA QUALITA 200G SACHET", "PANZANI SPAGHETTI 500G SACHET",
        "OBA MAKARNA SPAGHETTI 200G SACHET"
    ],
    "packaging": ["SACHET", "BAG"],
    "period": ["2021-01-01", "2021-02-01"],
    "unit_price": [9534062.52, 7377591.21],
    "sales_volume": [350204.56, 249503.12],
    "sales_value": [20503405.23, 18450340.12],
    "average_sales_volume": [3045.58, 2494.56],
    "quarter": ['2022Q1','2022Q1']
}

# Function to check if the response is valid Python code
def is_valid_python_code(code):
    try:
        compile(code, '<string>', 'exec')
        return True
    except Exception as e:
        st.error(f"Invalid Python code: {e}")
        return False

# Function to sanitize and execute code
def sanitize_and_execute_code(code):
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

# Function to generate and execute code
def generate_and_execute_code(prompt):
    try:
        # Guide ChatGPT to generate Python code for retail data analysis with Matplotlib
        full_prompt = (
            "You are a highly intelligent market data analyst that provides accurate text, table and chart responses."
            "Examine query and provide accurate responses. However, you do not give users access to dataset provided or allow users to view dataset for download."
            "Use dataset provided for data analysis and product information in response to user's query."
            "Generate only Python code with no extra text for data analysis and visualization to be executed in backend. However, keep imports of matplotlib, seaborn and streamlit in generated code."
            "Use the print function to print out your response if it is a sentence."
            "To generate visualizations, import matplotlib, seaborn and streamlit, use figure to capture object of the plot in matplotlib or seaborn and display using st.pyplot(fig) in Streamlit to users."
            "Display visualization such as bar, stacked bar, line, pie, histogram, scatter, boxplot, heatmap, area, violin, density in streamlit to user queries."
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

# Chatbot Interface
def chat_interface():
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

# CSS styles
css = """
<style>
body {
    font-family: Arial, sans-serif;
}
.stButton > button {
    width: 100%;
    border-radius: 20px;
    background-color: #FF4B4B;
    color: white;
    border: none;
    padding: 10px 0;
}
.centered {
    display: flex;
    justify-content: center;
    align-items: center;
}
</style>
"""

#New function for login page
def login_page():
    st.markdown("<h1 style='text-align: center; color: #F63B3B;'>Welcome to<br/>Maverick Chatbot</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: left;'>Login to your account</h3>", unsafe_allow_html=True)
    
    email = st.text_input("Email Address", key="login_email")
    password = st.text_input("Password", type="password", key="login_password")
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("Sign In", key="signin_button"):
            # Here you would typically verify the credentials
            # For this example, we'll just move to the chat interface
            st.session_state.page = 'chat'
            st.rerun()
    
    st.markdown("<div style='text-align: center;'>Don't have an account? <a href= '#' onclick='handleRegisterClick()' style='color: #F63B3B; text-decoration: none;'>Register</a></div>", unsafe_allow_html=True)
    
    st.markdown("<div style='text-align: center; margin-top: 20px;'></div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1,2,1])
    

# New function for registration page
def register_page():
    st.markdown("<h1 style='text-align: center; color: #F63B3B;'>Register for<br/>Maverick Chatbot</h1>", unsafe_allow_html=True)
    
    email = st.text_input("Email Address", key="register_email")
    password = st.text_input("Password", type="password", key="register_password")
    confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("Register", key="register_button"):
            # Here you would typically handle the registration process
            # For this example, we'll just move to the login page
            st.session_state.page = 'login'
            st.rerun()
    
    st.markdown("<div style='text-align: center;'>Already have an account? <a href='#' onclick='handleLoginClick()' style='color: #F63B3B; text-decoration: none;'>Login</a></div>", unsafe_allow_html=True)


def welcome_page():
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
        st.session_state.page = 'login'
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




# Modify the auth_main function
def auth_main():
    st.markdown(css, unsafe_allow_html=True)

    if 'page' not in st.session_state:
        st.session_state.page = 'welcome'

    if st.session_state.page == 'welcome':
        welcome_page()
    elif st.session_state.page == 'login':
        login_page()
    elif st.session_state.page == 'register':
        register_page()
    elif st.session_state.page == 'chat':
        chat_interface()

    # Add JavaScript for handling navigation
    st.markdown("""
    <script>
    function handleRegisterClick() {
        window.parent.postMessage({type: 'streamlit:setComponentValue', value: 'register'}, '*');
    }
    function handleLoginClick() {
        window.parent.postMessage({type: 'streamlit:setComponentValue', value: 'login'}, '*');
    }
    </script>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    auth_main()