import boto3
import streamlit as st
import time

# --- AWS CONFIGURATION (Using Streamlit Secrets) ---
try:
    AWS_ACCESS_KEY = st.secrets["AWS_ACCESS_KEY_ID"]
    AWS_SECRET_KEY = st.secrets["AWS_SECRET_ACCESS_KEY"]
    REGION = st.secrets.get("AWS_DEFAULT_REGION", "us-east-1")
except KeyError:
    st.error("AWS Secrets not found. Please configure them in the Streamlit Dashboard or .streamlit/secrets.toml.")
    st.stop()

# Your specific Resource IDs
S3_BUCKET = "rag-pipeline-nithisha-2026"
KNOWLEDGE_BASE_ID = st.secrets["KNOWLEDGE_BASE_ID"]
DATA_SOURCE_ID = st.secrets["DATA_SOURCE_ID"]

# Initialize AWS Session and Clients
session = boto3.Session(
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=REGION
)

s3_client = session.client('s3')
bedrock_agent = session.client('bedrock-agent')
bedrock_agent_runtime = session.client('bedrock-agent-runtime')

# --- MAIN UI SETUP ---
st.set_page_config(page_title="Policy Assistant", page_icon="📂")
st.title("📂 Policy Q&A Assistant")

# --- USER GUIDANCE ---
st.info("""
**What is this?** This is an AI Assistant designed to help you quickly find information within long policy manuals. 
Instead of searching through hundreds of pages, you can upload a document and simply ask questions to get instant answers.
\nFor example documents like SNAP, Medicaid, Health Insurance, Housing and Immigration policies work best, but feel free to try any PDF!

\n**Note:** This assistant can also understand and explain complex **tables, images, and flowcharts** that are often found in these technical documents.
""")

with st.expander("📖 How to use this assistant"):
    st.markdown("""
    1. **Step 1: Upload** – Go to the sidebar on the left and drag in a policy PDF.
    2. **Step 2: Sync** – Click **'Upload & Sync'**. This allows the Assistant to read and learn the new information.
    3. **Step 3: Wait** – Watch the status bar. Once it says **'Ready'**, you can begin.
    4. **Step 4: Ask** – Type your question in the chat box at the bottom (for example: *"Who is eligible for this program?"*).
    """)

# --- SIDEBAR: Upload & Sync ---
with st.sidebar:
    st.header("Manage Documents")
    st.write("Upload a new policy document below to update the Assistant's knowledge.")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if st.button("Upload & Sync"):
        if uploaded_file:
            s3_path = f"input-docs/{uploaded_file.name}"
            try:
                # 1. Upload to S3
                s3_client.upload_fileobj(uploaded_file, S3_BUCKET, s3_path)
                st.success(f"Uploaded {uploaded_file.name} to S3!")
                
                # 2. Start Ingestion Job
                with st.spinner("🔄 Assistant is reading the document..."):
                    response = bedrock_agent.start_ingestion_job(
                        knowledgeBaseId=KNOWLEDGE_BASE_ID,
                        dataSourceId=DATA_SOURCE_ID
                    )
                    job_id = response['ingestionJob']['ingestionJobId']

                    # 3. Poll for Completion
                    while True:
                        status_check = bedrock_agent.get_ingestion_job(
                            knowledgeBaseId=KNOWLEDGE_BASE_ID,
                            dataSourceId=DATA_SOURCE_ID,
                            ingestionJobId=job_id
                        )
                        status = status_check['ingestionJob']['status']
                        
                        if status == 'COMPLETE':
                            st.success("✅ Ready! You can now ask questions about this document.")
                            break
                        elif status == 'FAILED':
                            st.error("❌ The Assistant failed to read the document. Please check the AWS console.")
                            break
                        
                        time.sleep(5) 
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.error("Please select a PDF file first.")
            
# --- MAIN CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your Policy Assistant. Once you've uploaded a document in the sidebar, feel free to ask me anything about it!"}
    ]

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input and Retrieval
if prompt := st.chat_input("Ask a question about policy..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            response = bedrock_agent_runtime.retrieve_and_generate(
                input={'text': prompt},
                retrieveAndGenerateConfiguration={
                    'type': 'KNOWLEDGE_BASE',
                    'knowledgeBaseConfiguration': {
                        'knowledgeBaseId': KNOWLEDGE_BASE_ID,
                        'modelArn': 'us.anthropic.claude-3-5-sonnet-20241022-v2:0'
                    }
                }
            )
            answer = response['output']['text']
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            st.error(f"An error occurred: {e}")