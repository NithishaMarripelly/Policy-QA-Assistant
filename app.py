import boto3
import streamlit as st
import time

# --- AWS CONFIGURATION (Using Streamlit Secrets) ---
# This pulls credentials from your .streamlit/secrets.toml (local) 
# or the Streamlit Cloud Dashboard (deployed)
try:
    AWS_ACCESS_KEY = st.secrets["AWS_ACCESS_KEY_ID"]
    AWS_SECRET_KEY = st.secrets["AWS_SECRET_ACCESS_KEY"]
    REGION = st.secrets.get("AWS_DEFAULT_REGION", "us-east-1")
except KeyError:
    st.error("AWS Secrets not found. Please configure them in the Streamlit Dashboard or .streamlit/secrets.toml.")
    st.stop()

# Your specific Resource IDs
S3_BUCKET = "rag-pipeline-nithisha-2026"
KNOWLEDGE_BASE_ID = "SKEHRRDGDA"
DATA_SOURCE_ID = "GPKSPDVNBO"

# Initialize AWS Session and Clients
session = boto3.Session(
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=REGION
)

s3_client = session.client('s3')
bedrock_agent = session.client('bedrock-agent')
bedrock_agent_runtime = session.client('bedrock-agent-runtime')

st.set_page_config(page_title="Policy-RAG Assistant", page_icon="📂")
st.title("📂 Policy-RAG Assistant")

# --- SIDEBAR: Document Management ---
with st.sidebar:
    st.header("Manage Documents")
    uploaded_file = st.file_uploader("Upload a new policy PDF", type="pdf")
    
    if st.button("Upload & Sync"):
        if uploaded_file:
            # 1. Upload the file to your S3 bucket
            s3_path = f"input-docs/{uploaded_file.name}"
            try:
                s3_client.upload_fileobj(uploaded_file, S3_BUCKET, s3_path)
                st.success(f"Uploaded {uploaded_file.name} to S3!")
                
                # 2. Trigger and Poll the Ingestion Job (Sync)
                with st.spinner("🔄 AI is reading and indexing your documents..."):
                    response = bedrock_agent.start_ingestion_job(
                        knowledgeBaseId=KNOWLEDGE_BASE_ID,
                        dataSourceId=DATA_SOURCE_ID
                    )
                    job_id = response['ingestionJob']['ingestionJobId']

                    # Check status until complete
                    while True:
                        status_check = bedrock_agent.get_ingestion_job(
                            knowledgeBaseId=KNOWLEDGE_BASE_ID,
                            dataSourceId=DATA_SOURCE_ID,
                            ingestionJobId=job_id
                        )
                        status = status_check['ingestionJob']['status']
                        
                        if status == 'COMPLETE':
                            st.success("✅ Sync Complete! Your AI is now updated.")
                            break
                        elif status == 'FAILED':
                            reason = status_check['ingestionJob'].get('failureReasons', ['Unknown error'])
                            st.error(f"❌ Sync failed: {reason}")
                            break
                        
                        time.sleep(5) # Wait to avoid API throttling
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.error("Please select a PDF file first.")

# --- MAIN CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input and RAG Retrieval
if prompt := st.chat_input("Ask a question about policy..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Assistant Response
    with st.chat_message("assistant"):
        try:
            response = bedrock_agent_runtime.retrieve_and_generate(
                input={'text': prompt},
                retrieveAndGenerateConfiguration={
                    'type': 'KNOWLEDGE_BASE',
                    'knowledgeBaseConfiguration': {
                        'knowledgeBaseId': KNOWLEDGE_BASE_ID,
                        # Using the cross-region inference profile ID to avoid capacity errors
                        'modelArn': 'us.anthropic.claude-3-5-sonnet-20241022-v2:0'
                    }
                }
            )
            answer = response['output']['text']
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            st.error(f"An error occurred: {e}")