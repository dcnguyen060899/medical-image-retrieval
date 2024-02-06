import streamlit as st
import os
import sys
import openai

# initialize open ai api keys and deeplake
openai.api_key = st.secrets["openai_api_key"]
# os.environ["ACTIVELOOP_TOKEN"] = ''
# Fetching secrets
os.environ['ACTIVELOOP_TOKEN'] = st.secrets["active_loop_token"]

# %%
# Imports
#
from typing import List

from llama_index import (    Document,
    ServiceContext,
    SimpleDirectoryReader,
    VectorStoreIndex,
)
from llama_index.agent import OpenAIAgent
from llama_index.llms import OpenAI
from llama_index.multi_modal_llms import OpenAIMultiModal
from llama_index.output_parsers import PydanticOutputParser
from llama_index.program import MultiModalLLMCompletionProgram
from llama_index.tools import FunctionTool, QueryEngineTool, ToolMetadata
from llama_index.vector_stores import DeepLakeVectorStore
from pydantic import BaseModel

from openai_utils import generate_image_description
from llama_index.readers.deeplake import DeepLakeReader
import random
from llama_index.storage.storage_context import StorageContext

from typing import List, Tuple
import deeplake
from PIL import Image
from io import BytesIO
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from llama_index import set_global_service_context
from llama_index import ServiceContext, VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings import OpenAIEmbedding


def retrieve_image(image_id, folder):

  """Data QA select a folder, each image name from the folder will retrieve by the query_engine tool."""

  ds = deeplake.load(f'hub://dcnguyen060899/{folder}')
  image_ids = [id_list[0] for id_list in ds['ids'].numpy()]

  index = image_ids.index(image_id)
  image_data = ds['images'][index].numpy()
  image_data = st.image(image_data, channels="BGR")

  return ds, image_id

def retrieve_image_id(folder):
  ds = deeplake.load(f'hub://dcnguyen060899/{folder}')
  image_ids = [id_list[0] for id_list in ds['ids'].numpy()]

  return image_ids

retrieve_image_id_tool = FunctionTool.from_defaults(fn=retrieve_image_id)
retrieve_image_tool = FunctionTool.from_defaults(fn=retrieve_image)

# %%
# VectorStore connected to our DeepLake dataset
#



llm = OpenAI(model="gpt-4", temperature=0.7)

# %%
# Inventory query engine tool

agent = OpenAIAgent.from_tools(
  system_prompt = """
You are an advanced AI specifically trained in medical imaging retrieval from image vector database, with a focus on retrieving Data QA's image and its image_id.

Your tasked to provide CT scan images the Data QA upon request of folder and image_id.

Here is how the process work:
>>> Don't ask all the questions at once, gather the required information step by step.
>>> Firstly, most likely, the user would want to know what image_id we have in a specific folder (retrieve_image_id_tool). List all the image_id name available in the folder.
  >>> Always ask what folder if and only if the user did not provide.

>>> Once you have the required information, your answer needs to present image's description composed by the
image_id with the folder the image located.

>>> Note: the image_id corresponding to and ids. In the function: retrieve_image_id, it retrieves both the image_id and its ids. Provide that information for the user.
""",


  tools=[
        retrieve_image_tool,
        retrieve_image_id_tool,
    ],
    llm=llm,
    verbose=True)

st.title('👔 Medical Image Retrieval Engine 🧩')

# Session state for holding messages
if 'messages' not in st.session_state:
    st.session_state.messages = []
# Display past messages
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])
prompt = st.chat_input('Input your prompt here')
if prompt:
   # Directly query the OpenAI Agent
   st.chat_message('user').markdown(prompt)
   st.session_state.messages.append({'role': 'user', 'content': prompt})
   response = agent.chat(prompt)
   final_response = response.response
   st.chat_message('assistant').markdown(final_response)
   st.session_state.messages.append({'role': 'assistant', 'content': final_response})
