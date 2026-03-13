import boto3
import streamlit as st
import time
import uuid
import json

# --- 0. CUSTOM CSS FOR CHUNKS ---
st.markdown("""
    <style>
    /* 1. Target the Info Box specifically by its test-id */
    [data-testid="stInfo"] {
        background-color: #1e293b !important; /* Professional dark blue background */
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        padding: 15px !important;
    }

    /* 2. Force a clean, small font size on EVERY element inside the box */
    [data-testid="stInfo"] *,
    [data-testid="stInfo"] p,
    [data-testid="stInfo"] div,
    [data-testid="stInfo"] span {
        font-size: 12px !important;  /* Reduced from 14px to 12px */
        font-family: 'Inter', 'Source Sans Pro', sans-serif !important;
        line-height: 1.4 !important;
        color: #e2e8f0 !important; /* Soft white/grey for better readability */
        font-weight: normal !important;
        font-style: normal !important;
    }

    /* 3. Specifically fix the bold headers so they still stand out a little */
    [data-testid="stInfo"] strong, 
    [data-testid="stInfo"] b {
        font-size: 13px !important;  /* Slightly larger for headers */
        font-weight: 700 !important;
        color: #38bdf8 !important; /* Light blue for headings */
    }

    /* 4. Fix the expander header font weight */
    .st-ae {
        font-weight: 500;
    }
    
    /* 5. Target markdown content specifically */
    [data-testid="stInfo"] [data-testid="stMarkdownContainer"] {
        font-size: 12px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 1. SESSION INITIALIZATION ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# --- 2. AWS CONFIGURATION (Using Streamlit Secrets) ---
try:
    AWS_ACCESS_KEY = st.secrets["AWS_ACCESS_KEY_ID"]
    AWS_SECRET_KEY = st.secrets["AWS_SECRET_ACCESS_KEY"]
    REGION = st.secrets.get("AWS_DEFAULT_REGION", "us-east-1")
except KeyError:
    st.error("AWS Secrets not found. Please configure them in .streamlit/secrets.toml.")
    st.stop()

S3_BUCKET = "rag-pipeline-nithisha-2026"
KNOWLEDGE_BASE_ID = st.secrets["KNOWLEDGE_BASE_ID"]
DATA_SOURCE_ID = st.secrets["DATA_SOURCE_ID"]

# Initialize AWS Clients
session = boto3.Session(
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=REGION
)

s3_client = session.client('s3')
bedrock_agent = session.client('bedrock-agent')
bedrock_agent_runtime = session.client('bedrock-agent-runtime')

# --- 3. AUTOMATIC SESSION CLEANUP LOGIC ---
def auto_cleanup_callback(session_id_to_clean):
    try:
        s3 = boto3.client(
            's3',
            aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
            region_name=st.secrets.get("AWS_DEFAULT_REGION", "us-east-1")
        )
        prefix = f"input-docs/{session_id_to_clean}"
        objects = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)
        
        if 'Contents' in objects:
            delete_keys = [{'Key': obj['Key']} for obj in objects['Contents']]
            s3.delete_objects(Bucket=S3_BUCKET, Delete={'Objects': delete_keys})
    except Exception:
        pass 

@st.cache_resource(on_release=auto_cleanup_callback, scope="session")
def register_session_cleanup(session_id):
    return session_id

register_session_cleanup(st.session_state.session_id)

# --- 4. MAIN UI SETUP ---
st.set_page_config(page_title="Policy Assistant", page_icon="📂")
st.title("📂 Policy Q&A Assistant")

st.info("""
**What is this?** This is an AI Assistant designed to help you quickly find information within long policy manuals. 
Instead of searching through hundreds of pages, you can upload a document and simply ask questions to get instant answers.
\nFor example, documents like SNAP, Medicaid, Health Insurance, Housing, and Immigration policies work best.

\n**Note:** This assistant can also understand and explain complex **tables, images, and flowcharts** found in these technical documents.
""")

with st.expander("📖 How to use this assistant"):
    st.markdown("""
    1. **Step 1: Upload** – Go to the sidebar on the left and drag in a policy PDF.
    2. **Step 2: Sync** – Click **'Upload & Sync'**. This allows the Assistant to read and learn the new information.
    3. **Step 3: Wait** – Watch the status bar. Once it says **'Ready'**, you can begin.
    4. **Step 4: Ask** – Type your question in the chat box at the bottom.
    """)

# --- 5. SIDEBAR: Manage Documents ---
with st.sidebar:
    st.header("Manage Documents")
    st.write("Upload a new policy document below to update the Assistant's knowledge.")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if st.button("Upload & Sync"):
        if uploaded_file:
            file_key = f"{st.session_state.session_id}_{uploaded_file.name}"
            s3_path = f"input-docs/{file_key}"
            metadata_path = f"{s3_path}.metadata.json"
            
            metadata_content = {
                "metadataAttributes": {
                    "session_id": st.session_state.session_id
                }
            }
            
            try:
                s3_client.upload_fileobj(uploaded_file, S3_BUCKET, s3_path)
                s3_client.put_object(
                    Bucket=S3_BUCKET, 
                    Key=metadata_path, 
                    Body=json.dumps(metadata_content)
                )
                st.success(f"Pdf Uploaded {uploaded_file.name} Successfully!")
                
                with st.spinner("🔄 Assistant is reading the document..."):
                    response = bedrock_agent.start_ingestion_job(
                        knowledgeBaseId=KNOWLEDGE_BASE_ID,
                        dataSourceId=DATA_SOURCE_ID
                    )
                    job_id = response['ingestionJob']['ingestionJobId']

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
                            st.error("❌ The Assistant failed to read the document.")
                            break
                        time.sleep(5) 
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.error("Please select a PDF file first.")

# --- 6. MAIN CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your Policy Assistant. Once you've uploaded a document in the sidebar, feel free to ask me anything about it!"}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about policy..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # Retrieval with strict Session ID filtering
            response = bedrock_agent_runtime.retrieve_and_generate(
                input={'text': prompt},
                retrieveAndGenerateConfiguration={
                    'type': 'KNOWLEDGE_BASE',
                    'knowledgeBaseConfiguration': {
                        'knowledgeBaseId': KNOWLEDGE_BASE_ID,
                        'modelArn': 'us.anthropic.claude-3-5-sonnet-20241022-v2:0',
                        'retrievalConfiguration': {
                            'vectorSearchConfiguration': {
                                'filter': {
                                    'equals': {
                                        'key': 'session_id',
                                        'value': st.session_state.session_id
                                    }
                                }
                            }
                        }
                    }
                }
            )
            
            # Display main answer
            answer = response['output']['text']
            st.markdown(answer)

            # Display Citations (Top-K Chunks)
            if "citations" in response and response["citations"]:
                with st.expander("📚 View Source Chunks (Top-K)"):
                    chunk_count = 1
                    for citation in response["citations"]:
                        # NEW CODE - USE THIS:
                        for reference in citation.get("retrievedReferences", []):
                            source_text = reference["content"]["text"]
                            # Use HTML container with inline styles to force consistent sizing
                            st.markdown(f"""
                                <div style="
                                    background-color: #1e293b;
                                    border: 1px solid #334155;
                                    border-radius: 8px;
                                    padding: 15px;
                                    margin-bottom: 10px;
                                ">
                                    <div style="
                                        font-size: 13px;
                                        font-weight: 700;
                                        color: #38bdf8;
                                        margin-bottom: 8px;
                                    ">Source Chunk {chunk_count}:</div>
                                    <div style="
                                        font-size: 12px;
                                        color: #e2e8f0;
                                        line-height: 1.5;
                                        font-family: 'Inter', 'Source Sans Pro', sans-serif;
                                        white-space: pre-wrap;
                                    ">{source_text}</div>
                                </div>
                            """, unsafe_allow_html=True)
                            chunk_count += 1

            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            st.error(f"An error occurred: {e}")