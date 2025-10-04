import gradio as gr
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Function of loading the saved model components
def load_model():
    with open('cf_model.pkl', 'rb') as f:
        model_components = pickle.load(f)
    return model_components

# Loading the model components
model_components = load_model()
item_similarity = model_components['item_similarity']
movie_features = model_components['movie_features']
merged_db = model_components['merged_db']
item_final_ratings = model_components['item_final_ratings']

# Defining the recommendation function
def recommend_movies(user_id, N=5):
    """
    Recommends top N movies for a given user
    """
    try:
        user_id = int(user_id)
        # Get the predictions for this user
        user_predictions = item_final_ratings.loc[user_id]
        
        # Sort predictions in descending order and get top N
        top_n_recommendations = user_predictions.sort_values(ascending=False)[:N]
        
        # Get movie details and format output
        recommendations = []
        for movie_id, pred_rating in top_n_recommendations.items():
            movie_title = merged_db[merged_db['movieId'] == movie_id]['title'].iloc[0]
            recommendations.append(f"{movie_title} (ID: {movie_id}) - Predicted Rating: {pred_rating:.2f}")
        
        return "\n".join(recommendations)
    
    except KeyError:
        return f"User ID {user_id} not found in the dataset"
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Create the Gradio interface
iface = gr.Interface(
    fn=recommend_movies,
    inputs=[
        gr.Number(label="User ID", info="Enter a user ID to get movie recommendations"),
        gr.Slider(minimum=1, maximum=10, step=1, label="Number of Recommendations", value=5)
    ],
    outputs=gr.Textbox(label="Movie Recommendations"),
    title="Movie Recommendation System",
    description="This system uses Collaborative Filtering to recommend movies based on user preferences.",
    examples=[[120, 5], [36, 5], [42, 5]]
)

# Launch the interface
# type: python app.py into terminal to run the app
if __name__ == "__main__":
    iface.launch()