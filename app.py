import os
import traceback
import urllib.parse

import streamlit as st
import torch
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import sys

# Remove problematic torch.classes path from module watching
sys.modules['torch.classes'].__path__ = []

# -------------------- Streamlit Page Setup -------------------- #
st.set_page_config(page_title="AccessMate - Beauty Product Recommendation", page_icon="🌟", layout="wide")
st.title("🌟 AccessMate 🌟")
st.subheader("Intelligent Beauty Product Recommendation System")
st.write("Enter a prompt to get personalized recommendations for beauty products tailored for you!")

# -------------------- Session State -------------------- #
if 'prompt_history' not in st.session_state:
    st.session_state.prompt_history = []

# Sidebar Prompt History
st.sidebar.title("📝 Prompt History")
for i, (prompt, response) in enumerate(reversed(st.session_state.prompt_history[-5:]), 1):
    with st.sidebar.expander(f"Prompt {i}"):
        st.write(f"**Prompt**: {prompt}")
        st.write(f"**Response**: {response}")

# -------------------- Helper Functions -------------------- #
def get_secret(key, default=None):
    try:
        return st.secrets.get(key, default)
    except Exception as e:
        st.error(f"Error accessing secret '{key}': {e}")
        return default

def connect_to_mongodb():
    try:
        mongo_uri = st.secrets["mongodb"]["uri"]
        password = st.secrets["api"]["key"]

        if not mongo_uri:
            st.error("MongoDB URI missing in secrets.toml.")
            st.stop()

        if "{password}" in mongo_uri:
            encoded_password = urllib.parse.quote_plus(str(password))
            mongo_uri = mongo_uri.replace("{password}", encoded_password)

        client = MongoClient(mongo_uri)
        client.admin.command('ping')

        db = client["cosmetic_app_db"]
        collection = db["cosmetic_app_embeddings"]
        return client, db, collection
    except Exception as e:
        st.error(f"MongoDB connection failed: {e}")
        st.stop()
        return None, None, None

def load_embedding_model():
    try:
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        st.error(f"Embedding model load failed: {e}")
        st.stop()

def setup_gemini_api():
    try:
        api_key = get_secret('api')
        if not api_key:
            st.error("Gemini API key missing.")
            st.stop()
        genai.configure(api_key=api_key)
        return genai.GenerativeModel("gemini-2.0-flash")
    except Exception as e:
        st.error(f"Gemini setup failed: {e}")
        st.stop()

# def get_product_recommendations(prompt, model, collection):
#     try:
#         query_vector = model.encode(prompt, convert_to_tensor=False)
#         if not isinstance(query_vector, list):
#             query_vector = query_vector.tolist()

#         pipeline = [
#             {"$vectorSearch": {
#                 "index": "vector_index_cosmetic_db",
#                 "queryVector": query_vector,
#                 "path": "embedding",
#                 "exact": True,
#                 "limit": 3
#             }},
#             {"$project": {
#                 "_id": 0,
#                 "id": 1,
#                 "brand": 1,
#                 "name": 1,
#                 "image_link": 1,
#                 "product_link": 1,
#                 "text_full_description": 1,
#                 "score": {"$meta": "vectorSearchScore"}
#             }}
#         ]
#         return list(collection.aggregate(pipeline))
#     except Exception as e:
#         st.error(f"Error getting recommendations: {e}")
#         return []


def get_product_recommendations(prompt, model, collection):
    try:
        query_vector = model.encode(prompt, convert_to_tensor=False)
        if not isinstance(query_vector, list):
            query_vector = query_vector.tolist()

        pipeline = [
            {"$vectorSearch": {
                "index": "vector_index_cosmetic_db",
                "queryVector": query_vector,
                "path": "embedding",
                "exact": True,
                "limit": 3
            }},
            {"$project": {
                "_id": 0,
                "id": 1,
                "brand": 1,
                "name": 1,
                "image_link": 1,
                "product_link": 1,
                "text_full_description": 1,
                "score": {"$meta": "vectorSearchScore"}
            }}
        ]
        return list(collection.aggregate(pipeline))
    except Exception as e:
        st.error(f"Error getting recommendations: {e}")
        st.text(traceback.format_exc())  # Add traceback for debugging
        return []
# Function to display product recommendations

def display_products(results):
    recommendations = []
    for product in results:
        st.write(f"### {product.get('name', 'Unnamed')} ({product.get('brand', 'Unknown')})")
        if image_url := product.get("image_link"):
            st.image(image_url, width=200)
        st.write(product.get("text_full_description", "No description"))
        st.write(f"**Score:** {product.get('score', 0):.2f}")
        st.markdown(f"[🌐 View Product]({product.get('product_link', '#')})", unsafe_allow_html=True)
        st.write("---")

        recommendations.append(
            f"Product: {product.get('name')}\n"
            f"Brand: {product.get('brand')}\n"
            f"Description: {product.get('text_full_description')}\n"
            f"Score: {product.get('score'):.2f}\n"
        )
    return recommendations

def get_ai_recommendation(prompt, recommendations, model):
    if len(recommendations) < 3:
        return "Not enough matching products to generate a detailed comparison."
    
    gen_prompt = f"""
    Based on the following product recommendations for the prompt "{prompt}", suggest the best product and justify your choice:

    {recommendations[0]}
    {recommendations[1]}
    {recommendations[2]}
    """

    try:
        response = model.generate_content(gen_prompt)
        return response.text if response else "No response from Gemini."
    except Exception as e:
        return f"Error generating AI recommendation: {e}"

# -------------------- Main Application Logic -------------------- #
def main():
    client, db, collection = connect_to_mongodb()
    embed_model = load_embedding_model()
    gemini_model = setup_gemini_api()

    st.selectbox("Select a category:", ["Product Recommendation"])
    prompt = st.text_input("Enter your prompt:")

    if st.button("Get Recommendations"):
        if not prompt:
            st.warning("Prompt cannot be empty.")
            return

        with st.spinner("Fetching recommendations..."):
            try:
                results = get_product_recommendations(prompt, embed_model, collection)

                if not results:
                    st.warning("No matching products found.")
                    return

                recommendations = display_products(results)
                ai_recommendation = get_ai_recommendation(prompt, recommendations, gemini_model)

                st.write("### Recommended Product and Reasoning")
                st.write(ai_recommendation)

                st.session_state.prompt_history.append((prompt, ai_recommendation))
                if len(st.session_state.prompt_history) > 5:
                    st.session_state.prompt_history.pop(0)

            except Exception as err:
                st.error("An error occurred while processing your request.")
                st.error(traceback.format_exc())

# -------------------- Entry Point -------------------- #
if __name__ == "__main__":
   main()




