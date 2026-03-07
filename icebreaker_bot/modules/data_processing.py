import json
import config

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex

from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def split_profile_data(profile_data):

    profile_text = json.dumps(profile_data)

    document = Document(text=profile_text)

    splitter = SentenceSplitter(chunk_size=config.CHUNK_SIZE)

    nodes = splitter.get_nodes_from_documents([document])

    return nodes


def create_vector_database(nodes):

    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    index = VectorStoreIndex(nodes, embed_model=embed_model)

    return index
