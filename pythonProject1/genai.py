import streamlit as st
from transformers import pipeline

# Cache models
@st.cache_resource
def load_qa_model():
    return pipeline("text2text-generation", model="google/flan-t5-base")

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

@st.cache_resource
def load_ner_model():
    return pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)

st.set_page_config(page_title="GenAI Healthcare Assistant", layout="centered")
st.title("🧬 GenAI Healthcare Assistant")
st.write("This app can answer medical questions, summarize reports, and extract medical terms.")

# Sidebar for task selection
task = st.sidebar.selectbox(
    "Choose a task:",
    ("Medical Q&A", "Summarize Medical Report", "Extract Medical Entities")
)

# Load models
qa_model = load_qa_model()
summarizer = load_summarizer()
ner_model = load_ner_model()

if task == "Medical Q&A":
    st.subheader("🤖 Ask a Medical Question")
    query = st.text_area("Enter your question:")
    if st.button("Get Answer"):
        if query:
            with st.spinner("Generating answer..."):
                result = qa_model(f"Answer the medical question: {query}", max_length=200)
                st.success(result[0]['generated_text'])
        else:
            st.warning("Please enter a question.")

elif task == "Summarize Medical Report":
    st.subheader("📄 Summarize a Medical Report")
    report = st.text_area("Paste the medical report here:")
    if st.button("Summarize"):
        if report:
            with st.spinner("Summarizing..."):
                summary = summarizer(report, max_length=130, min_length=30, do_sample=False)
                st.success(summary[0]['summary_text'])
        else:
            st.warning("Please paste the report.")

elif task == "Extract Medical Entities":
    st.subheader("🔎 Extract Drug Names, Conditions, and Tests")
    text = st.text_area("Paste a medical paragraph here:")
    if st.button("Extract Entities"):
        if text:
            with st.spinner("Extracting..."):
                entities = ner_model(text)
                for ent in entities:
                    st.markdown(f"- **{ent['entity_group']}**: {ent['word']} (score: {ent['score']:.2f})")
        else:
            st.warning("Please enter some text.")

