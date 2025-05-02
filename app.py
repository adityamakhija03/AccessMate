import os
import streamlit as st
import google.generativeai as genai
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import urllib.parse
import torch
import traceback

# Set page configuration
st.set_page_config(
    page_title="AccessMate - Beauty Product Recommendation",
    page_icon="🌟",
    layout="wide"
)

# Streamlit App Title and Description
st.title("🌟 AccessMate 🌟")
st.subheader("Intelligent Beauty Product Recommendation System")
st.write("Enter a prompt to get personalized recommendations for beauty products tailored for you!")

# Initialize session state for prompt history
if 'prompt_history' not in st.session_state:
    st.session_state.prompt_history = []

# Sidebar for Prompt History
st.sidebar.title("📝 Prompt History")
if st.session_state.prompt_history:
    for i, (hist_prompt, hist_response) in enumerate(reversed(st.session_state.prompt_history[-5:]), 1):
        with st.sidebar.expander(f"Prompt {i}"):
            st.write(f"**Prompt**: {hist_prompt}")
            st.write(f"**Response**: {hist_response}")

# Function to safely load secrets
def get_secret(key, default=None):
    try:
        return st.secrets.get(key, default)
    except Exception as e:
        st.error(f"Error accessing secret '{key}': {e}")
        return default

# Function to safely connect to MongoDB
def connect_to_mongodb():
    try:
        # Get MongoDB credentials
        mongo_uri = st.secrets["mongodb"]["uri"]
        password = api_key = st.secrets["api"]["key"]
        
        if not mongo_uri:
            st.error("MongoDB URI not found in secrets. Please check your .streamlit/secrets.toml file.")
            st.stop()
            
        # If password needs to be added to URI
        if password and "{password}" in mongo_uri:
            # Ensure password is a string before encoding
            password_str = str(password)
            encoded_password = urllib.parse.quote_plus(password_str)
            mongo_uri = mongo_uri.replace("{password}", encoded_password)
        
        # Connect to MongoDB
        client = MongoClient(mongo_uri)
        # Test connection
        client.admin.command('ping')
        
        # Access database and collection
        db = client["cosmetic_app_db"]
        collection = db["cosmetic_app_embeddings"]
        
        return client, db, collection
    
    except Exception as e:
        st.error(f"Error connecting to MongoDB: {str(e)}")
        st.error("Please check your MongoDB credentials in the secrets.toml file.")
        st.stop()
        return None, None, None

# Function to safely load SentenceTransformer model
def load_embedding_model():
    try:
        # device = torch.device('cpu')
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return model
    except Exception as e:
        st.error(f"Error loading SentenceTransformer model: {str(e)}")
        st.stop()
        return None

# Function to safely configure Gemini API
def setup_gemini_api():
    try:
        api_key = get_secret('api')
        if not api_key:
            st.error("Gemini API key not found in secrets. Please check your .streamlit/secrets.toml file.")
            st.stop()
            
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        return model
    except Exception as e:
        st.error(f"Error setting up Gemini API: {str(e)}")
        st.stop()
        return None

# Function to get product recommendations
def get_product_recommendations(prompt, embedding_model, collection):
    try:
        # Generate query embedding
        query_embedding = embedding_model.encode(prompt, convert_to_tensor=False)
        
        # MongoDB query for product recommendations based on embeddings
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index_cosmetic_db",
                    "queryVector": query_embedding if isinstance(query_embedding, list) else query_embedding.tolist(),
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
        return results
    except Exception as e:
        st.error(f"Error generating recommendations: {str(e)}")
        return []

# Function to display product recommendations
def display_products(results):
    top_recommendations = []
    
    for result in results:
        st.write(f"### {result.get('name', 'No name')} ({result.get('brand', 'No brand')})")
        
        # Display product image
        image_link = result.get('image_link', '')
        if image_link:
            try:
                st.image(image_link, width=200)
            except Exception as img_err:
                st.warning(f"Could not load image: {str(img_err)}")
        
        # Display product details
        st.write(result.get('text_full_description', 'No description'))
        st.write(f"**Relevance Score:** {result.get('score', 0):.2f}")
        st.markdown(f"[🌐 View Product]({result.get('product_link', '#')})", unsafe_allow_html=True)
        st.write("---")
        
        # Create recommendation text
        recommendation = (
            f"Product: {result.get('name', 'No name')}\n"
            f"Brand: {result.get('brand', 'No brand')}\n"
            f"Description: {result.get('text_full_description', 'No description')}\n"
            f"Score: {result.get('score', 0):.2f}\n"
        )
        top_recommendations.append(recommendation)
    
    return top_recommendations

# Function to get AI recommendation using Gemini
def get_ai_recommendation(prompt, top_recommendations, gemini_model):
    try:
        if len(top_recommendations) < 3:
            return "Not enough matching products to generate a detailed comparison."
            
        gemini_prompt = f"""
        Based on the following product recommendations for the prompt "{prompt}", suggest the best recommendation and provide reasoning:

        {top_recommendations[0]}
        {top_recommendations[1]}
        {top_recommendations[2]}

        Which product would you recommend, and why?
        """

        response = gemini_model.generate_content(gemini_prompt)
        if response:
            return response.text
        else:
            return "No response from the Gemini model."
    except Exception as e:
        return f"Error generating AI recommendation: {str(e)}"

# Main function
def main():
    # Setup connections and models
    client, db, collection = connect_to_mongodb()
    embedding_model = load_embedding_model()
    gemini_model = setup_gemini_api()
    
    # Category selection and user input for prompt
    category = st.selectbox("Select a category:", ["Product Recommendation"])
    prompt = st.text_input("Enter your prompt:")

    # Generate Recommendations
    if st.button("Get Recommendations"):
        if not prompt:
            st.warning("Please enter a prompt before requesting recommendations.")
            return
            
        with st.spinner("Generating recommendations..."):
            try:
                # Get product recommendations
                results = get_product_recommendations(prompt, embedding_model, collection)
                
                if not results:
                    st.warning("No matching products found. Please try a different query.")
                    return
                
                # Display products
                top_recommendations = display_products(results)
                
                # Get AI recommendation
                ai_recommendation = get_ai_recommendation(prompt, top_recommendations, gemini_model)
                
                # Display AI recommendation
                st.write("### Recommended Product and Reasoning")
                st.write(ai_recommendation)
                
                # Save prompt and response to session history
                st.session_state.prompt_history.append((prompt, ai_recommendation))
                if len(st.session_state.prompt_history) > 5:
                    st.session_state.prompt_history.pop(0)
                    
            except Exception as e:
                st.error(f"Error processing your request: {str(e)}")
                st.error(f"Detailed error: {traceback.format_exc()}")

# Run the app
if __name__ == "__main__":
    main()