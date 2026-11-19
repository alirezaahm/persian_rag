"""

# 14050823 | Saturday


- running 
''' python -m streamlit run main.py '''

- is equivalent to:
''' streamlit run main_v1.py '''

-check: in this version, ui is complete.

"""

##
import streamlit as st 
import pandas as pd
from PIL import Image 
import base64
from pathlib import Path 
from rag_v1 import PersianRAG
from openai import OpenAI
import gc
import subprocess
from llama_manager import LlamaServerManager
import os 
import signal



@st.cache_resource
def get_llama_manager():
    return LlamaServerManager()

llama_manager = get_llama_manager()


## show image background
persian_art_image = Image.open("images/2.jpg")
st.set_page_config(page_title="Persian-RAG", page_icon=persian_art_image, layout="wide")


## show title
original_title = '<p style="font-family:Courier; color:white; font-size: 50px; margin-top: -20px; margin-bottom: 0px; line-height: 1; margin-left: 0px;">:::: PERSIAN RAG ::::</p>'
st.markdown(original_title, unsafe_allow_html=True)


## show text
line_title = '<p style="font-family:Courier; color:white; font-size: 40px; margin-top: -30px; margin-bottom: 0px; line-height: 1; margin-left: 12px;">___________________</p>'
st.markdown(line_title, unsafe_allow_html=True)

line1_right = 10 # 0 = left, bigger = more right
line1_down = -20 # negative = up, positive = down
line2_right = 10
line2_down = 0

# line2 first, line1 second = swapped
st.markdown(f"""
<p style="margin-left: {line2_right}px; margin-top: {line2_down}px; color: white;">
Credit: Design, engineering, and implement by <a href="https://www.linkedin.com/in/alireza-ahmadi-7ba990196/">Alireza Ahmadi</a>.
</p>
""", unsafe_allow_html=True)

st.markdown(f"""
<p style="margin-left: {line1_right}px; margin-top: {line1_down}px; color: white;">
This web application is for use RAG on Persian documentations.
</p>
""", unsafe_allow_html=True)

line_title_1 = '<p style="font-family:Courier; color:white; font-size: 40px; margin-top: -50px; margin-bottom: 0px; line-height: 1; margin-left: 12px;">___________________</p>'
st.markdown(line_title_1, unsafe_allow_html=True)


def get_server_models(base_url):
	try:
		client = OpenAI(base_url=base_url,api_key="local")
		response = client.models.list()
		models = [model.id for model in response.data]
		return models

	except Exception:
		return []

## sidebar: LLM, EMBEDDING AND RETRIEVAL CONFIGURATION
MODEL_OPTIONS = {
    "Gemma": {
        "Gemma 3 1B": {
            "ollama": "gemma3:1b",
            "alias": "gemma3:1b",
            "path": "/home/alireza/models/gemma-3-1b-it-Q4_K_M.gguf"
        },
        "Gemma 3 4B": {
            "ollama": "gemma3:4b",
            "alias": "gemma3:4b",
            "path": "/home/alireza/models/gemma-3-4b-it-Q4_K_M.gguf"
        },
    },

    "Liquid AI (LFM)": {
        "LFM2.5 350M": {
            "ollama": None,
            "alias": "lfm2.5-350m",
            "path": "/home/alireza/AI_Models/LFM2.5-350M-Q4_K_M.gguf"
        },
        "LFM2.5 1.2B Thinking": {
            "ollama": None,
            "alias": "lfm2.5-1.2b",
            "path": "/home/alireza/AI_Models/LFM2.5-1.2B-Thinking-Q4_K_M.gguf"
        },
        "LFM3 1.7B": {
            "ollama": None,
            "alias": "lfm3-1.7b",
            "path": "/home/alireza/AI_Models/LFM3-1.7B-Q4_K_M.gguf"
        },
    },
}


with st.sidebar:
    st.header("AI Model Settings")
    # llm configuration
    with st.expander("LLM Configuration", expanded=True):

        llm_provider = st.selectbox("LLM Framework:",["Ollama", "llama.cpp"],key="llm_provider")
        # only show the model family supported by the provider.
        if llm_provider == "Ollama":
            available_families = ["Gemma"]
        else:
            available_families = ["Liquid AI (LFM)"]

        model_family = st.selectbox("Model Family:",available_families,key="model_family")
        selected_name = st.selectbox(
            "LLM Model:",
            list(MODEL_OPTIONS[model_family].keys()),
            key="llm_model_selection")

        selected_model = MODEL_OPTIONS[model_family][selected_name]

        if llm_provider == "Ollama":
            llm_url = "http://localhost:11434/v1"
            llm_model = selected_model["ollama"]
            llm_model_path = None

        else:
            llm_url = "http://localhost:8080/v1"
            llm_model = selected_model["alias"]

            llm_model_path = st.text_input(
                "GGUF Model Path:",
                value=selected_model["path"],
                key="llm_model_path"
            )

            if not Path(llm_model_path).is_file():
                st.error("The selected GGUF model file does not exist.")

    # embedding configuration
    with st.expander("Embedding Configuration", expanded=True):
        embedding_provider = st.selectbox(
            "Embedding Provider:",
            ["SentenceTransformers", "llama.cpp API"],
            key="embedding_provider"
        )

        if embedding_provider == "SentenceTransformers":
            embedding_model_name = st.selectbox(
                "Embedding Model:",
                ["intfloat/multilingual-e5-small"],
                key="embedding_model"
            )

            embedding_url = ""

        else:
            embedding_url = st.text_input(
                "Embedding API URL:",
                value="http://localhost:8081/v1",
                key="embedding_url"
            )
            embedding_model_name = st.text_input(
                "Embedding Model ID:",
                value="multilingual-e5-small",
                key="embedding_model_id"
            )

    # retrieval configuration
    with st.expander("Retrieval Settings", expanded=True):
        top_k = st.slider(
            "Number of retrieved chunks:",
            min_value=1,
            max_value=15,
            value=5,
            key="retrieval_top_k"
        )

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = """
    <style>
    body {
            background-image: url("data:image/jpg;base64, %s");
            background-size: cover;
            
    }
    </style>
    """ % bin_str

    st.markdown(page_bg_img, unsafe_allow_html=True)

set_png_as_page_bg("images/1.jpg")

## show wallpaper
def get_base64_of_bin_wall_file(jpg_file):
    with open(jpg_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_wall(jpg_file):
    bin_str = get_base64_of_bin_wall_file(jpg_file)
    st.markdown("""
    <style>
    .stApp {
    background: url("your-image-url-here");
    background-size: cover;
    }
    </style>""", unsafe_allow_html=True)

set_png_as_page_wall("images/1.jpg")

## Create or reuse RAG object
rag_config = (llm_model,llm_url,embedding_provider,embedding_model_name,embedding_url)
if ("rag" not in st.session_state or st.session_state.get("rag_config") != rag_config):
    # release old embedding model if one exists
    old_rag = st.session_state.get("rag")

    if old_rag is not None:
        old_rag.release_embedding_model()

    # create new RAG object
    st.session_state["rag"] = PersianRAG(
        llm_model=llm_model,
        llm_url=llm_url,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model_name,
        embedding_url=embedding_url
    )

    st.session_state["rag_config"] = rag_config

rag = st.session_state["rag"]


def shutdown_persian_rag():
    print("Starting Persian RAG shutdown...")
    # 1. release the embedding model
    rag.release_embedding_model()
    # 2. release document data from application RAM
    rag.chunks = []
    rag.embeddings = None
    rag.last_retrieved = []
    # 3. stop llama.cpp server
    try:
        llama_manager.stop()
        print("llama.cpp server stopped.")
    except Exception as error:
        print("llama.cpp shutdown error:", error)

    # 4. unload all currently running Ollama models
    try:
        result = subprocess.run(["ollama", "ps"],capture_output=True,text=True,timeout=15)
        lines = result.stdout.strip().splitlines()
        # first line is the table header
        for line in lines[1:]:
            parts = line.split()
            if not parts:
                continue

            model_name = parts[0]
            subprocess.run(["ollama", "stop", model_name],capture_output=True,text=True,timeout=30)
            print("Ollama model unloaded:", model_name)

    except Exception as error:
        print("Ollama shutdown error:", error)

    # 5. clear Streamlit session data
    st.session_state.clear()
    # 6. release Python garbage
    gc.collect()
    print("Persian RAG resources released.")
    # 7. terminate the Streamlit process
    os.kill(os.getpid(), signal.SIGTERM)


## buttons
# shut down / turn off / release memory
if st.button("Exit / Turn Off", type="primary"):
    shutdown_persian_rag()

# reset button
move_right_button1 = 0 # 0 = left, bigger = more right. Try 0, 100, 300
move_down_button1 = -20 # negative = up, positive = down. Try -100, -50, 0, 50
st.markdown(f"""
<style>
div[data-testid="stButton"] {{
    margin-left: {move_right_button1}px !important;
    margin-top: {move_down_button1}px !important;
}}
</style>
""", unsafe_allow_html=True)


if st.button("Reset"):
    st.write("Reset everything in program.")
    rag.clear_index()
    rag.chunks = []
    rag.release_embedding_model()
    llama_manager.stop()
    if llm_provider == "Ollama":
    	subprocess.run(["ollama", "stop", llm_model],capture_output=True,text=True)
    rag.embeddings = None
    rag.last_retrieved = []
    st.session_state["document_signature"] = None
    st.session_state["answer"] = ""
    st.success("Saved document index deleted.")
    # refresh()
    # streamlit_js_eval(streamlit_js_eval="parent.window.location.reload()")
    st.rerun() # reload (refresh) the webpage


## uploader wedgit
st.sidebar.subheader("Upload documents")
# csv uploader 
uploaded_files_csv = st.sidebar.file_uploader(
	"Upload CSV data",
	accept_multiple_files=True,
	type="csv",
)
for uploaded_file_csv in uploaded_files_csv:
	df = pd.read_csv(uploaded_file_csv)
	st.write(df)	

# pdf uploader
uploaded_files_pdf = st.sidebar.file_uploader(
	"Upload PDF or DOCX files:",
	accept_multiple_files=True,
	type=["pdf", "docx"],
	# key=f"pdf_uploader_{st.session_state['uploader_version']}"
)
documents = []
if uploaded_files_pdf:
	for uploaded_file_pdf in uploaded_files_pdf:
		documents.append((uploaded_file_pdf.name, uploaded_file_pdf.getvalue()))


## document indexing
document_signature = tuple((name, len(content), hash(content))
							for name, content in documents)

document_signature = tuple(
    (name, len(content), hash(content))
    for name, content in documents)

if st.session_state.get("document_signature") != document_signature:
    if documents:
        try:
            with st.spinner("Reading and indexing documents..."):
                number_of_chunks = rag.add_documents(documents)

            st.sidebar.success(f"Indexed {number_of_chunks} chunks.")

        except Exception as error:
            st.error(f"Indexing error: {error}")

    else:
        rag.clear_index()
    st.session_state["document_signature"] = document_signature

if rag.index_warning:
    st.warning(rag.index_warning)
elif rag.chunks:
    st.sidebar.success(f"Loaded {len(rag.chunks)} saved chunks.")


## user text input 
input_right = 70 # 0 = left, bigger = more right. Try 0, 100, 300
input_down = -100 # negative = up, positive = down. Try -50, 50
input_width = 800 # smaller = narrower box. Try 400, 600, 800

st.markdown(f"""
<style>
div[data-testid="stTextInput"] {{
	margin-left: {input_right}px !important;
	margin-top: {input_down}px !important;
	max-width: {input_width}px !important;
}}
<style>

""", unsafe_allow_html=True)

# user_question = st.text_input("Question:", type="search")
user_question = st.text_input("Question:", placeholder ="Ask your question in Persian...")
submit = st.button("Ask")


## generate answer
if submit:
	if not user_question.strip():
		st.warning("Plese enter a question.")
	elif not rag.chunks:
		st.warning("Please upload documents first.")
	else:
		try: 
			if llm_provider == "llama.cpp":
				with st.spinner("Searching and generating answer..."):
					llama_manager.start(
						model_path=llm_model_path,
						model_alias = llm_model)
					# answer = rag.ask(user_question, top_k=top_k)

				# st.session_state["answer"] = answer
			with st.spinner("Searching and generating answer..."):
				answer = rag.ask(user_question, top_k=top_k)
			st.session_state["answer"] = answer

		except Exception as error:
			st.error(f"Error: {error}")


## display answer
if st.session_state.get("answer"):
	st.markdown("----")
	st.subheader("Answer")
	st.write(st.session_state["answer"])


## display sources
if rag.last_retrieved:
	with st.expander("View retrieved document chunks"):
		for result in rag.last_retrieved:
			st.markdown(f"Document: {result.chunk.source_document}")
			st.markdown(f"Location: {result.chunk.source}")
			st.markdown(f"Similarity {result.score:.4f}")
			st.write(result.chunk.text)
			st.markdown("----")





