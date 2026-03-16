import boto3
import streamlit as st
import time
import uuid
import json
import html
import io
import base64
import fitz  # PyMuPDF for image extraction

# --- 0. CUSTOM CSS FOR CHUNKS & UI ---
st.markdown("""
    <style>
    /* Custom styling for chunk containers */
    .chunk-container {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
    }
    
    .chunk-header {
        font-size: 13px;
        font-weight: 700;
        color: #38bdf8;
        margin-bottom: 10px;
        font-family: 'Inter', sans-serif;
    }
    
    .chunk-content {
        font-size: 12px;
        line-height: 1.6;
        color: #e2e8f0;
        font-family: 'Inter', 'Source Sans Pro', sans-serif;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    
    /* Ensure expander content doesn't override */
    [data-testid="stExpander"] {
        background-color: transparent;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background-color: #38bdf8;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 1. SESSION INITIALIZATION ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# --- 2. AWS CONFIGURATION ---
try:
    AWS_ACCESS_KEY = st.secrets["AWS_ACCESS_KEY_ID"]
    AWS_SECRET_KEY = st.secrets["AWS_SECRET_ACCESS_KEY"]
    REGION = st.secrets.get("AWS_DEFAULT_REGION", "us-east-1")
except KeyError as e:
    st.error(f"Configuration error: {e} not found in secrets.toml")
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
bedrock_runtime = session.client('bedrock-runtime')  # For vision API calls

# --- 3. IMAGE EXTRACTION & DESCRIPTION FUNCTIONS ---

def extract_images_from_pdf(pdf_bytes):
    """
    Extract images from PDF using PyMuPDF (fitz)
    Returns: List of dicts with {page_num, image_index, image_bytes, image_format}
    """
    images = []
    
    try:
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            image_list = page.get_images(full=True)
            
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                # Only process images larger than 10KB (filter out small icons/logos)
                if len(image_bytes) > 10000:
                    images.append({
                        "page_num": page_num + 1,
                        "image_index": img_index + 1,
                        "image_bytes": image_bytes,
                        "image_format": image_ext
                    })
        
        pdf_document.close()
        return images
        
    except Exception as e:
        st.warning(f"Error extracting images: {e}")
        return []


def describe_image_with_bedrock(image_bytes, image_format, page_num):
    """
    Use Amazon Bedrock's Claude with vision to describe an image
    Returns: String description of the image
    """
    try:
        # Convert image bytes to base64
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Determine media type
        media_type_map = {
            "png": "image/png",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "webp": "image/webp",
            "gif": "image/gif"
        }
        media_type = media_type_map.get(image_format.lower(), "image/png")
        
        # Prepare the request body for Bedrock
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_base64
                            }
                        },
                        {
                            "type": "text",
                            "text": """Analyze this image from a policy document. Provide a detailed description that includes:

1. **Type of content**: Is it a table, chart, flowchart, diagram, form, or illustration?
2. **Key information**: What data, process, or concept does it show?
3. **Specific details**: Include any numbers, labels, categories, steps, or relationships shown
4. **Context**: What policy-related information would someone need from this image?

Be thorough but concise. Focus on searchable, factual content."""
                        }
                    ]
                }
            ]
        }
        
        # Call Bedrock Runtime API with Claude 3.5 Sonnet
        response = bedrock_runtime.invoke_model(
            modelId='us.anthropic.claude-3-5-sonnet-20241022-v2:0',
            body=json.dumps(request_body)
        )
        
        # Parse response
        response_body = json.loads(response['body'].read())
        description = response_body['content'][0]['text']
        
        return description
        
    except Exception as e:
        return f"[Image description unavailable: {str(e)}]"


def create_image_description_file(image_descriptions):
    """
    Create a text file with all image descriptions for vector search
    """
    text_content = "=== IMAGE DESCRIPTIONS FOR VECTOR SEARCH ===\n\n"
    text_content += "This document contains AI-generated descriptions of all images, charts, tables, "
    text_content += "diagrams, and flowcharts found in the policy document. These descriptions enable "
    text_content += "searching for visual content.\n\n"
    
    for desc in image_descriptions:
        text_content += f"--- Page {desc['page_num']}, Image {desc['image_index']} ---\n"
        text_content += f"{desc['description']}\n\n"
    
    return text_content.encode('utf-8')


# --- 4. AUTOMATIC SESSION CLEANUP LOGIC ---
def auto_cleanup_callback(session_id_to_clean):
    try:
        s3 = boto3.client('s3', 
            aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
            region_name=st.secrets.get("AWS_DEFAULT_REGION", "us-east-1"))
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

# --- 5. MAIN UI SETUP ---
st.set_page_config(page_title="Policy Assistant", page_icon="📂")
st.title("📂 Policy Q&A Assistant")

st.info("""
**What is this?** This is an AI Assistant designed to help you quickly find information within long policy manuals. Instead of searching through hundreds of pages, you can upload a document and simply ask questions to get instant answers.

For example, documents like SNAP, Medicaid, Health Insurance, Housing, and Immigration policies work best.

**Note:** This assistant can now **understand and explain complex tables, images, flowcharts, and diagrams** by automatically analyzing them with AI vision technology.
""")

with st.expander("📖 How to use this assistant"):
    st.markdown("""
    1. **Step 1: Upload** – Go to the sidebar on the left and drag in a policy PDF.
    2. **Step 2: Processing** – The system will automatically extract and analyze all images in the document using AWS Bedrock.
    3. **Step 3: Sync** – Click **'Upload & Sync'**. This allows the Assistant to read and learn the new information.
    4. **Step 4: Wait** – Watch the status bar. Once it says **'Ready'**, you can begin.
    5. **Step 5: Ask** – Type your question in the chat box at the bottom, including questions about charts, tables, and diagrams!
    """)

# --- 6. SIDEBAR: Manage Documents ---
with st.sidebar:
    st.header("Manage Documents")
    st.write("Upload a new policy document below to update the Assistant's knowledge.")
    
    # Add toggle for image processing
    enable_image_processing = st.checkbox(
        "Enable Image Analysis (recommended for documents with charts/diagrams)",
        value=True,
        help="When enabled, the system will extract and analyze all images using AI vision"
    )
    
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if st.button("Upload & Sync"):
        if uploaded_file:
            # Read the uploaded file
            pdf_bytes = uploaded_file.read()
            uploaded_file.seek(0)  # Reset file pointer
            
            file_key = f"{st.session_state.session_id}_{uploaded_file.name}"
            s3_path = f"input-docs/{file_key}"
            metadata_path = f"{s3_path}.metadata.json"
            
            try:
                image_descriptions = []
                
                # Step 1: Extract and describe images (if enabled)
                if enable_image_processing:
                    with st.spinner("🔍 Extracting images from PDF..."):
                        images = extract_images_from_pdf(pdf_bytes)
                        
                        if images:
                            st.info(f"Found {len(images)} images in the document")
                            
                            # Step 2: Describe images with Bedrock Claude Vision
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            for idx, img_data in enumerate(images):
                                status_text.text(f"🤖 Analyzing image {idx + 1} of {len(images)} (Page {img_data['page_num']})...")
                                
                                description = describe_image_with_bedrock(
                                    img_data["image_bytes"],
                                    img_data["image_format"],
                                    img_data["page_num"]
                                )
                                
                                image_descriptions.append({
                                    "page_num": img_data["page_num"],
                                    "image_index": img_data["image_index"],
                                    "description": description
                                })
                                
                                progress_bar.progress((idx + 1) / len(images))
                            
                            status_text.empty()
                            progress_bar.empty()
                            st.success(f"✅ Successfully analyzed {len(images)} images using AWS Bedrock")
                        else:
                            st.info("No significant images found in this document")
                
                # Step 3: Create companion text file with image descriptions
                if image_descriptions:
                    with st.spinner("📝 Creating searchable image descriptions..."):
                        description_text = create_image_description_file(image_descriptions)
                        description_file_key = f"{st.session_state.session_id}_{uploaded_file.name}_image_descriptions.txt"
                        description_s3_path = f"input-docs/{description_file_key}"
                        
                        # Upload companion text file
                        s3_client.put_object(
                            Bucket=S3_BUCKET,
                            Key=description_s3_path,
                            Body=description_text,
                            Metadata={
                                'session_id': st.session_state.session_id,
                                'source_document': uploaded_file.name,
                                'image_count': str(len(image_descriptions))
                            }
                        )
                        
                        # Also add metadata file for the description file
                        desc_metadata_path = f"{description_s3_path}.metadata.json"
                        s3_client.put_object(
                            Bucket=S3_BUCKET, 
                            Key=desc_metadata_path, 
                            Body=json.dumps({"metadataAttributes": {"session_id": st.session_state.session_id}})
                        )
                
                # Step 4: Upload original PDF
                with st.spinner("📤 Uploading document to cloud..."):
                    s3_client.upload_fileobj(io.BytesIO(pdf_bytes), S3_BUCKET, s3_path)
                    s3_client.put_object(
                        Bucket=S3_BUCKET, 
                        Key=metadata_path, 
                        Body=json.dumps({"metadataAttributes": {"session_id": st.session_state.session_id}})
                    )
                    st.success(f"✅ Uploaded {uploaded_file.name}")
                
                # Step 5: Start ingestion job
                with st.spinner("🔄 Assistant is indexing the document and image descriptions..."):
                    job = bedrock_agent.start_ingestion_job(
                        knowledgeBaseId=KNOWLEDGE_BASE_ID, 
                        dataSourceId=DATA_SOURCE_ID
                    )
                    job_id = job['ingestionJob']['ingestionJobId']
                    
                    while True:
                        status = bedrock_agent.get_ingestion_job(
                            knowledgeBaseId=KNOWLEDGE_BASE_ID, 
                            dataSourceId=DATA_SOURCE_ID, 
                            ingestionJobId=job_id
                        )['ingestionJob']['status']
                        
                        if status == 'COMPLETE':
                            success_msg = "✅ Ready! You can now ask questions about this document"
                            if image_descriptions:
                                success_msg += f" and its {len(image_descriptions)} analyzed images."
                            else:
                                success_msg += "."
                            st.success(success_msg)
                            break
                        elif status == 'FAILED':
                            st.error("❌ The Assistant failed to read the document.")
                            break
                        time.sleep(5)
                        
            except Exception as e:
                st.error(f"Error: {e}")
                import traceback
                st.error(traceback.format_exc())
        else:
            st.error("Please select a PDF file first.")

# --- 7. MAIN CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm your Policy Assistant. Once you've uploaded a document in the sidebar, feel free to ask me anything about it - including questions about charts, tables, and diagrams!"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

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
                        'modelArn': 'us.anthropic.claude-3-5-sonnet-20241022-v2:0',
                        'retrievalConfiguration': {
                            'vectorSearchConfiguration': {
                                'filter': {'equals': {'key': 'session_id', 'value': st.session_state.session_id}}
                            }
                        }
                    }
                }
            )
            
            answer = response['output']['text']
            st.markdown(answer)

            # Display Citations with small, consistent font
            if "citations" in response and response["citations"]:
                with st.expander("📚 View Source Chunks (Top-K)"):
                    chunk_count = 1
                    for citation in response["citations"]:
                        for reference in citation.get("retrievedReferences", []):
                            source_text = reference.get("content", {}).get("text", "Text content not found for this chunk.")
                            
                            # Escape HTML to prevent markdown interpretation
                            escaped_text = html.escape(source_text)
                            
                            # Display using HTML with fixed small font size
                            st.markdown(f"""
                                <div class="chunk-container">
                                    <div class="chunk-header">Source Chunk {chunk_count}:</div>
                                    <div class="chunk-content">{escaped_text}</div>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            chunk_count += 1

            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            st.error(f"An error occurred: {e}")