# 📂 Policy-QA Assistant

**Policy-QA Assistant** is a Retrieval-Augmented Generation (RAG) application that democratizes access to complex government policy manuals (like SNAP and Medicaid). By combining the power of **Amazon Bedrock** with a user-friendly **Streamlit** interface, this tool allows users to upload dense policy documents and get instant, accurate answers in plain English.

---

## 🚀 Features
- **Dynamic PDF Upload**: Users can upload new policy documents directly to an S3 bucket from the UI.
- **Automated Data Sync**: Triggers a Bedrock Knowledge Base ingestion job automatically upon upload to keep the AI "brain" updated.
- **Context-Aware Chat**: Uses Claude 3.5 Sonnet to generate responses grounded strictly in the provided policy context.
- **Secure Infrastructure**: Built using AWS IAM best practices and Streamlit Secrets for credential management.

---

## 🏗️ Technical Architecture



1. **Frontend**: Streamlit (Python)
2. **Orchestration**: Amazon Bedrock Knowledge Bases
3. **Data Storage**: Amazon S3 (PDFs) and Amazon OpenSearch Serverless (Vector Embeddings)
4. **AI Models**: 
   - **Parsing/Generation**: Anthropic Claude 3.5 Sonnet
   - **Embeddings**: Amazon Titan Text Embeddings V2

---

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.9+
- AWS Account with Bedrock Model Access (Claude 3.5 Sonnet & Titan)
- A configured Bedrock Knowledge Base

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/Policy-QA-Assistant.git](https://github.com/your-username/Policy-QA-Assistant.git)
cd Policy-QA-Assistant
