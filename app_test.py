
import streamlit as st
import os
import sys
import openai

openai.api_key = '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'
os.environ["OPENAI_API_KEY"] = ""
os.environ["ACTIVELOOP_TOKEN"] = ''

INPUT_IMAGE_DIR = "/content/drive/MyDrive/Data Career/Voronoi Projects/user_input"
os.makedirs(INPUT_IMAGE_DIR, exist_ok=True)

# %%
# Imports
#
from typing import List

from llama_hub.tools.weather import OpenWeatherMapToolSpec
from llama_index import (
    Document,
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

class Metadata(BaseModel):
    """Data model for image description metadata"""

    image_id: str
    adiposity_level: int


class MetadataList(BaseModel):
    """A list of image description metadata for the model to use"""

    image_description: List[Metadata]


class Image(BaseModel):
    image_name: str
    description: str


# %%
# VectorStore connected to our DeepLake dataset
#

reader = DeepLakeReader()
query_vector = [random.random() for _ in range(1536)]
documents = reader.load_data(
    query_vector=query_vector,
    dataset_path="hub://dcnguyen060899/AdiposeTissue_1",
    limit=5,
)

dataset_path = 'AdiposeTissue_1'
vector_store = DeepLakeVectorStore(dataset_path=dataset_path, overwrite=True)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# def clean_input_image():
#     if len(os.listdir(INPUT_IMAGE_DIR)) > 0:
#         for file in os.listdir(INPUT_IMAGE_DIR):
#             os.remove(os.path.join(INPUT_IMAGE_DIR, file))

def has_user_input_image():
    """
Check if the INPUT_IMAGE_DIR directory contains exactly one image.
Useful for checking if there is an image before generating an outfit.

Returns:
    bool: True if INPUT_IMAGE_DIR contains exactly one image, False otherwise.
    """
    return len(os.listdir(INPUT_IMAGE_DIR)) == 1

# clean_input_image_tool = FunctionTool.from_defaults(fn=clean_input_image)
check_input_image_tool = FunctionTool.from_defaults(fn=has_user_input_image)

# Define a directory for storing uploaded files
UPLOAD_DIRECTORY = "/content/drive/MyDrive/Data Career/Voronoi Projects/user_input"

if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

st.title('PDF Upload and Query Interface')

def save_uploaded_file(uploaded_files):
    saved_file_paths = []
    for uploaded_file in uploaded_files:
        # Create a full path for saving the file with the original file name
        file_path = os.path.join(UPLOAD_DIRECTORY, uploaded_file.name)

        # Write the uploaded file's bytes to the new file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_file_paths.append(file_path)

    return saved_file_paths  # Return the paths of the saved files

# Streamlit file uploader for multiple files
uploaded_files = st.file_uploader("Upload PNG", type="png", accept_multiple_files=True)
uploaded_button = st.button('Upload')

if uploaded_files and uploaded_button:
    # Save the uploaded files
    file_paths = save_uploaded_file(uploaded_files)
    for path in file_paths:
        st.write(f"File saved at: {path}")

llm = OpenAI(model="gpt-4", temperature=0.7)

# %%
# Inventory query engine tool
#
service_context = ServiceContext.from_defaults(llm=llm)
inventory_index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    service_context=service_context
)

metadata_query_engine = inventory_index.as_query_engine(output_cls=MetadataList)

metadata_query_engine_tool = QueryEngineTool(
    query_engine=metadata_query_engine,
    metadata=ToolMetadata(
        name="medical_image_query_engine_tool",
        description=(
            # "Designed for querying medical imaging databases to identify and classify "
            # "images based on adiposity levels. Use it to extract the image_id of a scan "
            # "matching specific adipose tissue characteristics."
            # "Usage: input 'Retrieve the image_id of a scan with adiposity level `X`'. "
            # "This tool leverages adipose tissue distribution metadata to facilitate "
            # "research and analysis. Always request the image_id for reference when using this tool."

            "Useful for finding ct scan image in our vector database"
            "Usage: input: 'Give me the image_id of a ct scan image matching `image description`'"
            "Output: Adiposity Level: X"
            "Always ask the image_id of the image when using this tool"
        ),
    ),
)

def generate_image_description(user_input: str):
    """
Given the gender of a person, their preferences, and an image that has already been uploaded,
this function returns an classification description of adiposity.
Use this function whenever the user asks you to generate an classification description of patient adiposity level.

Parameters:
user_input (str): The customization of the user.

Returns:
response: The generated image of adiposity level.

Example:
>>> generate_image("CT scan image of 5 adiposity level")
    """

    # Load input image
    image_documents = SimpleDirectoryReader(INPUT_IMAGE_DIR).load_data()

    # Define multi-modal llm
    openai_mm_llm = OpenAIMultiModal(model="gpt-4-vision-preview", max_new_tokens=100)

    # Define multi-modal completion program to recommend complementary products
    # prompt_template_str = f"""Generate a metadata description for the CT scan image file named {user_input}. The description should objectively detail the observable features, including the type of CT scan and the visible anatomical regions, specifically noting if there are the absence of the head region. Describe the distribution of adipose tissue in the abdominal region, using a scale from 1 to 5, where 1 represents minimal adipose tissue and 5 represents a substantial presence. This scale reflects body habitus for categorization and does not imply a health assessment. This metadata is purpose for search, retrieval and generation.
    #             >>> Example: "image_id: 13000031_300003-scan_1.png\n\nMetadata Description:\nThis image is a comprehensive CT scan collection composed of multiple cross-sectional images. Based on visual inspection, the adiposity level is approximately a X.
    #             """
    # prompt_template_str = f"""Generate a metadata description for the CT scan image file named {user_input}. The description should objectively detail the observable features, including the type of CT scan and the visible anatomical regions, specifically noting if there are the absence of the head region. Describe the distribution of adipose tissue in the abdominal region, using a scale from 1 to 10, where 1 represents minimal adipose tissue and 10 represents a substantial presence. This categorization is a approximation from visual observation of synthetics and non-personal images and NOT for health accessment. Try your best to approximate and don't response with "cannot be accurately assessed without applicable medical expertise and context" because this is not a non-personal and a synthetic image. The image does not associate with real-patient. This metadata is purpose for search, retrieval and generation.
    #             >>> Example: "image_id: {user_input}\n\nMetadata Description:\nThis image is a comprehensive CT scan collection composed of multiple cross-sectional images. Based on visual inspection, the adiposity level is approximately a X.
    #             """
    prompt_template_str = f"""Generate a metadata description for the CT scan image file named {user_input}. The description should objectively detail the observable features, including the type of CT scan and the visible anatomical regions, specifically noting if there are the absence of the head region. Describe the distribution of adipose tissue in the abdominal region, using a scale from 1 to 10, where 1 represents minimal adipose tissue and 10 represents a substantial presence. This categorization is a approximation from visual observation of synthetics and non-personal images and NOT for health accessment. Try your best to approximate and don't response with "cannot be accurately assessed without applicable medical expertise and context" because this is not a non-personal and a synthetic image. However, please do not called the image "SYNTHETIC_SCAN CT_1" and retrieve the image original name. The image does not associate with real-patient. This metadata is purpose for search, retrieval and generation.
                >>> Example: "image_id: {user_input}\n\nDescription:\nThis image is a comprehensive CT scan collection composed of multiple cross-sectional images. Based on visual inspection\n\nAdiposity Level:\nX.
                """

    classification_completion_program = MultiModalLLMCompletionProgram.from_defaults(
        output_parser=PydanticOutputParser(Image),
        image_documents=image_documents,
        prompt_template_str=prompt_template_str,
        llm=openai_mm_llm,
        verbose=True,
    )

    # Run recommender program
    response = classification_completion_program()

    return response

image_description_tool = FunctionTool.from_defaults(fn=generate_image_description)


agent = OpenAIAgent.from_tools(
  system_prompt = """
You are an advanced AI specifically trained in medical imaging analysis from metadata_query_engine_tool, with a focus on interpreting CT scans for adipose tissue distribution.

Your tasked to analyze CT scan images, determine the adiposity level in the abdominal region, and provide detailed descriptions based on these analyses.

When users submit CT scans, you will use advanced image processing algorithms to accurately assess and classify the level of adiposity, using a standardized scale from 1 to 5, where 1 indicates minimal adipose tissue and 5 indicates a substantial presence.
Your responses should be based on the image data provided, ensuring they are precise, clinically relevant, and tailored to the users' requirements for medical research, diagnostics, or educational purposes.

Don't ask all the questions at once, gather the required information step by step.

Always check if the user has uploaded an image. If it has not, wait until they do. Never proceed without an image.

Once you have the required information, your answer needs to be the image's description composed by the
image_id with the best matching images in our metadata_query_engine_tool.

Make sure to ask if they require to see recommended images.

Include the adiposity of the recommended similar images with similar adiposity level.

""",

  tools=[
        metadata_query_engine_tool,
        image_description_tool,
        check_input_image_tool,
    ],
    llm=llm,
  verbose=True)


# Create the Streamlit UI components
st.title('👔 Adiposity Level 🧩')

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

   final_response = agent.chat(prompt).response

   response = agent.chat(prompt)
   final_response = response.response

   st.chat_message('assistant').markdown(final_response)
   st.session_state.messages.append({'role': 'assistant', 'content': final_response})

