"""

# 14050825 | Monday


---- pipeline:

pdf doc -> text extraction -> text cleaning -> chunking
-> embedding -> index storage (save chunks and vectors 
in json) | user asks question -> question embedding ->
similarity search -> retrieve top chunks -> llm -> fina
answer


Application starts
       |
       v
Embedding model = None
       |
       v
User uploads PDF
       |
       v
create_embedding()
       |
       v
Load embedding model into RAM
       |
       v
Generate embeddings

"""

import gc 
import re
import unicodedata
from io import BytesIO
from dataclasses import dataclass, asdict
import numpy as np 
import docx 
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import json 
import os 
from pathlib import Path 


@dataclass(frozen=True)
class Chunk:
    id:str 
    text:str 
    normalized_text:str
    source:str 
    source_document:str 

@dataclass(frozen=True)
class RetrievedChunk:
    chunk:Chunk
    score:float


## main rag class
class PersianRAG:
    def __init__(
        self,
        # llm_provider="ollama",
        llm_model="gemma3:1b",
        llm_url="http://localhost:11434/v1",
        embedding_provider="SentenceTransformers",
        embedding_model="intfloat/multilingual-e5-small",
        embedding_url="http://localhost:8081/v1",
        index_dir="data/persian_rag_index"
    ):

        # llm configuration
        self.llm_model = llm_model
        self.client = OpenAI(base_url=llm_url, api_key="local")
        self.embedding_provider = embedding_provider
        self.embedding_model_name = embedding_model
        self.embedding_url = embedding_url
        # keep the embedding model unloaded until it is needed
        self.embedding_model = None
        self.embedding_client = None
        # local embedding model
        if self.embedding_provider == "SentenceTransformers":
            pass

        # api embedding
        elif self.embedding_provider == "llama.cpp API":
            self.embedding_client = OpenAI(base_url=self.embedding_url,api_key="local")

        else:
            raise ValueError(f"Unknown embedding provider: {self.embedding_provider!r}")

        # persitent storage directory
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_path = self.index_dir / "chunks.jsonl"
        self.embeddings_path = self.index_dir / "embeddings.npy"
        self.confign_path = self.index_dir  / "index_config.json"
        # store document chunks and vectors 
        self.chunks = []
        self.embeddings = None
        # store last retrieval results
        self.last_retrieved = []
        self.index_warning = None
        # load previously saved index
        self.load_index()

    def get_embedding_config(self):
        return {
            "provider": self.embedding_provider,
            "model": self.embedding_model_name,
            "url": self.embedding_url
        }

    def load_index(self):
        self.chunks = []
        self.embeddings = None 
        self.index_warning = None 
        # check whether saved files exist
        if not self.chunks_path.exists() or not self.embeddings_path.exists():
            return 
        # check which embedding model created this index
        if self.confign_path.exists():
            with open(self.confign_path, "r", encoding="utf-8") as file:
                saved_config = json.load(file)

            if saved_config != self.get_embedding_config():
                self.index_warning = (
                    "the saved index was created using a different "
                    "embedding configuration. please re-index your documents."
                    )
                return 

        # load chunk metadata and text 
        with open(self.chunks_path, "r", encoding="utf-8") as file:
            for line in file:
                data = json.loads(line) # converts each JSON line back into a Python dictionary.
                self.chunks.append(Chunk(**data)) # reconstructs the original Chunk object from that dictionary.

        # load embeddings without reading the whole file into ram
        self.embeddings = np.load(self.embeddings_path, mmap_mode="r")
        #check consistency
        if len(self.chunks) != len(self.embeddings):
            self.chunks = []
            self.embeddings = None
            self.index_warning = "saved chunks and embeddings do not match"

    def clear_index(self):
        # clear chunks and vectors from ram
        self.chunks=[]
        self.embeddings=None 
        self.last_retrieved = []
        # delete saved files
        for path in [
                    self.chunks_path, 
                    self.embeddings_path, 
                    self.confign_path,
                    self.index_dir / "chunks.tmp.jsonl",
                    self.index_dir / "embeddings.tmp.npy",
        ]:
            if path.exists():
                path.unlink()

        gc.collect()
        print("Document index cleared.")

    ## persian normalization
    def normalize_persian(self, text):
        # ????
        text = unicodedata.normalize("NFKC", text) 
        replacements = {
            "ك": "ک",
            "ي": "ی",
            "ى": "ی",
            "ۀ": "ه",
            "ة": "ه",
            "أ": "ا",
            "إ": "ا",
            "٠": "0",
            "١": "1",
            "٢": "2",
            "٣": "3",
            "٤": "4",
            "٥": "5",
            "٦": "6",
            "٧": "7",
            "٨": "8",
            "٩": "9",
            "۰": "0",
            "۱": "1",
            "۲": "2",
            "۳": "3",
            "۴": "4",
            "۵": "5",
            "۶": "6",
            "۷": "7",
            "۸": "8",
            "۹": "9",
            "\u200c": " ",
            "\u200f": " ",
            "\u200e": " ",
            "ـ": ""
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        # remove arabic diacritics
        text = re.sub(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]", "", text)
        # remove unnecessary spaces
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    ## document ectraction
    def extract_document(self, filename, file_bytes):
        extention = filename.lower().split(".")[-1]
        text_blocks = []
        # read pdf 
        if extention == "pdf":
            pdf = PdfReader(BytesIO(file_bytes))
            for page_number, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    text_blocks.append((f"page:{page_number}", text))

        # read word document
        elif extention == "docx":
            document = docx.Document(BytesIO(file_bytes))
            for paragraph_number, paragraph in enumerate(document.paragraphs, start=1):
                if paragraph.text.strip():
                    text_blocks.append((f"paragraph:{paragraph_number}", paragraph.text))

        else:
            raise ValueError("only pdf and docs files are supported.")

        return text_blocks

    ## text chunking
    # if overlap_words >= max_words:
    #     raise ValueError("overlap_words must be smaller than max_words.")
    def create_chunks(self, text_blocks, filename, max_words=300, overlap_words=20):
        # check chunking parameters
        if max_words <= 0:
            raise ValueError("max_words must be greater than zero.")

        if overlap_words < 0 or overlap_words >= max_words:
            raise ValueError("overlap_words must be between 0 and max_words - 1.")

        chunks = []
        for source, text in text_blocks:
            words = text.split()
            start = 0
            while start < len(words):
                end = start + max_words
                selected_words = words[start:end]
                chunk_text = " ".join(selected_words)
                normalized_text = self.normalize_persian(chunk_text)

                if normalized_text:
                    chunk = Chunk(
                        id=f"chunk_{len(chunks) + 1}",
                        text = chunk_text,
                        normalized_text=normalized_text,
                        source=source,
                        source_document=filename
                    )
                    chunks.append(chunk)

                # move forward while keeping some overlap
                start += max_words - overlap_words

        return chunks 

    ## embedding generation
    def create_embedding(self, texts):
        # option 1- SentenceTransformers on cpu
        if self.embedding_provider == "SentenceTransformers":
            self.load_embedding_model()
            vectors = self.embedding_model.encode(
                texts,
                batch_size=2,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=True
            )
        # option 2- llama.cpp embedding api
        elif self.embedding_provider == "llama.cpp API":
            response = self.embedding_client.embeddings.create(
                model=self.embedding_model_name,
                input=texts,
                encoding_format="float"
            )
            # keep the original input order
            vectors = np.asarray([item.embedding for item in sorted(
                                response.data, key=lambda item: item.index
                                )],dtype=np.float32
            )

        else:
            raise ValueError("Unknown embedding provider.")

        # normalize vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.maximum(norms, 1e-12)
        return vectors.astype(np.float32)


    ## add documents
    def add_documents(self, documents):
        batch_size=2
        # [(filename, file_bytes), ...]
        all_chunks = []
        for filename, file_bytes in documents:
            text_blocks = self.extract_document(filename, file_bytes)
            document_chunks = self.create_chunks(text_blocks, filename)
            all_chunks.extend(document_chunks)

        if not all_chunks:
            raise ValueError(" no readable text found in the document.")

        number_of_chunks = len(all_chunks)
        print("total chunks:", number_of_chunks)

        # prepare temporary files
        temp_chunks_path = self.index_dir / "chunks.tmp.jsonl"
        temp_embeddings_path = self.index_dir / "embeddings.tmp.npy"

        # generate the first embedding batch
        first_batch = all_chunks[:batch_size]
        first_texts = [
            "passage: " + chunk.normalized_text 
            for chunk in first_batch]
        first_vectors = self.create_embedding(first_texts)
        embedding_dimension = first_vectors.shape[1]
        # create disk backed embedding matrix
        vector_store = np.lib.format.open_memmap(
            temp_embeddings_path,
            mode="w+",
            dtype=np.float32,
            shape=(number_of_chunks,embedding_dimension))
        # write the first batch 
        vector_store[:len(first_batch)] = first_vectors
        with open(temp_chunks_path, "w", encoding="utf-8") as file:
            for chunk in first_batch:
                json.dump(asdict(chunk),file,ensure_ascii=False)
                file.write("\n")
            # process remaining chunks in small batches
            for start in range(batch_size, number_of_chunks, batch_size):
                end = min(start + batch_size, number_of_chunks)
                batch_chunks = all_chunks[start:end]
                batch_texts = ["passage: " + chunk.normalized_text for chunk in batch_chunks]
                batch_vectors = self.create_embedding(batch_texts)
                vector_store[start:end] = batch_vectors
                # save chunk information
                for chunk in batch_chunks:
                    json.dump(asdict(chunk), file, ensure_ascii=False)
                    file.write("\n")

        # flush vectors to disk
        vector_store.flush()
        del vector_store
        # replace old index files with newfiles
        os.replace(temp_chunks_path, self.chunks_path)
        os.replace(temp_embeddings_path, self.embeddings_path)
        # save embedding configuration
        with open(self.confign_path, "w", encoding="utf-8") as file:
            json.dump(self.get_embedding_config(), file,ensure_ascii=False,indent=4)

        # load the new index from disk
        self.load_index()
        return number_of_chunks
        # # convert chunk texts into embedding vectors

    ## cosine similarity
    def calculate_similarity(self, question):
        if self.embeddings is None:
            raise ValueError("no embedding index available.")

        query_text = "query: " + self.normalize_persian(question)
        query_vector = self.create_embedding([query_text])[0]
        number_of_vectors = len(self.embeddings)
        scores = np.empty(number_of_vectors, dtype=np.float32)
        # compare vectors in small blocks
        block_size = 1024
        for start in range(0, number_of_vectors, block_size):
            end = min(start+block_size, number_of_vectors)
            vector_block = self.embeddings[start:end]
            scores[start:end] = vector_block @ query_vector

        return scores

    ## retrieval
    def retrieve(self, question, top_k=5):
        if not self.chunks:
            raise ValueError("please upload and index document first.")

        scores = self.calculate_similarity(question)
        # sort similarity scores from highest to lowest
        sorted_indexes = np.argsort(scores)[::-1]
        # select the best matching chunks
        selected_indexes = sorted_indexes[:top_k]
        retrieved_chunks = []
        for index in selected_indexes:
            result = RetrievedChunk(chunk=self.chunks[index], score=float(scores[index]))
            retrieved_chunks.append(result)

        self.last_retrieved = retrieved_chunks
        return retrieved_chunks

    ## generation
    def generate_answer(self, question, retrieved_chunks):
        context = ""
        for number, result in enumerate(retrieved_chunks, start=1):
            context += (
                f"\n[Source {number}: "
                f"{result.chunk.source_document}, "
                f"{result.chunk.source}]\n"
            )
            context += result.chunk.text + "\n"

        system_prompt = """
You are a Persian RAG assistant.
Answer the user's question using only the provided context.
If the context does not contain the answer, say:
پاسخ مناسب یافت نشد.

Answer in Persian(Farsi).
Do not invent information.
Be clear and concise.
"""
        
        user_prompt = f"""
Context:

{context}


Question:

{question}
"""

        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            temperature = 0.05, max_tokens = 4098
        )
        # debug
        print("---- LLM RESPONSE ----")
        print(response.model_dump_json(indent=2))
        print("----------------------")

        message = response.choices[0].message
        answer= message.content
        if not answer or not answer.strip():
            raise RuntimeError(
                "The LLM returned empty content. "
                f"Finish reason: {response.choices[0].finish_reason}. "
                f"Model: {self.llm_model}"
            )
        return answer.strip()

    ## complete rag pipeline
    def ask(self, question, top_k=5):
        # 1: retrieve relevant chunks
        retrieved_chunks = self.retrieve(question, top_k)
        # 2: generate answer 
        answer = self.generate_answer(question, retrieved_chunks)
        return answer

    def load_embedding_model(self):
        if self.embedding_provider == "SentenceTransformers":
            if self.embedding_model is None:
                self.embedding_model = SentenceTransformer(self.embedding_model_name,device="cpu")

    def release_embedding_model(self):
        if self.embedding_model is not None:
            del self.embedding_model
            self.embedding_model = None
        gc.collect()
        print("Embedding model released.")