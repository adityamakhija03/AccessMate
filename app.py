import os
import streamlit as st
import google.generativeai as genai
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import urllib.parse

# Streamlit App Title and Description
st.title("🌟 AccessMate 🌟")
st.title("|| Intelligent Beauty Product Recommendation System ||")
st.write("Enter a prompt to get personalized recommendations for Product, Catalog, or SEO needs. Discover the best options tailored for you!")

# Sidebar for Prompt History
if 'prompt_history' not in st.session_state:
    st.session_state.prompt_history = []

# Sidebar to display last five prompts and responses
st.sidebar.title("📝 Prompt History")
if st.session_state.prompt_history:
    for i, (hist_prompt, hist_response) in enumerate(reversed(st.session_state.prompt_history[-5:]), 1):
        with st.sidebar.expander(f"Prompt {i}"):
            st.write(f"**Prompt**: {hist_prompt}")
            st.write(f"**Response**: {hist_response}")

# Set up MongoDB client
encoded_password = urllib.parse.quote_plus(st.secrets['password'] )
uri = st.secrets['mongo_uri']
client = MongoClient(uri)
db = client["cosmetic_app_db"]
collection = db["cosmetic_app_embeddings"]

# Set up Generative AI (Gemini)
genai.configure(api_key= st.secrets['api'])
model = genai.GenerativeModel("gemini-1.5-flash")

# Load the Sentence Transformer Model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Select category and enter prompt
category = st.selectbox("Select a category:", ["Product Recommendation"])
prompt = st.text_input("Enter your prompt:")

if st.button("Get Recommendations"):
    # Generate embedding for the search query
    query_embedding = embedding_model.encode(prompt, convert_to_tensor=True)

    # MongoDB search pipeline for top 5 results based on embedding
    try:
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index_cosmetic_db",
                    "queryVector": query_embedding.tolist(),
                    "path": "embedding",
                    "exact": True,
                    "limit": 3
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "id": 1,
                    "brand": 1,
                    "image_link": 1,
                    "product_link": 1,
                    "name": 1,
                    "text_full_description": 1,
                    "score": {
                        "$meta": "vectorSearchScore"
                    }
                }
            }
        ]

        results = list(collection.aggregate(pipeline))

        # Gather and display top results with images and links
        top_recommendations = []
        for result in results:
            st.write(f"### {result.get('name', 'No name')} ({result.get('brand', 'No brand')})")
            st.image(result.get('image_link', ''), width=200)
            st.write(result.get('text_full_description', 'No description'))
            st.write(f"**Relevance Score:** {result.get('score', 'No score'):.2f}")
            st.markdown(f"[🌐 View Product]({result.get('product_link', '#')})", unsafe_allow_html=True)
            st.write("---")  # Divider for better readability

            recommendation = (
                f"Product: {result.get('name', 'No name')}\n"
                f"Brand: {result.get('brand', 'No brand')}\n"
                f"Description: {result.get('text_full_description', 'No description')}\n"
                f"Score: {result.get('score', 'No score')}\n"
            )
            top_recommendations.append(recommendation)

    except Exception as e:
        st.error(f"Error fetching data from MongoDB: {e}")

    # If results are sufficient, create prompt for Gemini API
    if len(top_recommendations) >= 3:
        gemini_prompt = f"""
        Based on the following product recommendations for the prompt "{prompt}", suggest the best recommendation and provide reasoning:

        {top_recommendations[0]}

        {top_recommendations[1]}

        {top_recommendations[2]}

        Which product would you recommend, and why?
        """

        # Get recommendation from Gemini
        try:
            response = model.generate_content(gemini_prompt)
            if response:
                st.write("### Recommended Product and Reasoning")
                st.write(response.text)

                # Save prompt and response in session history
                st.session_state.prompt_history.append((prompt, response.text))
                if len(st.session_state.prompt_history) > 5:  # Keep only the last 5 entries
                    st.session_state.prompt_history.pop(0)

            else:
                st.error("No response from the model.")
        except Exception as e:
            st.error(f"Error from Google Gemini API: {e}")
