import argparse
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

sample_captions = [
    "A colorful sunset over the mountains",
    "Children playing in a park on a sunny day",
    "A sleek modern kitchen with stainless steel appliances",
    "A cat napping on a windowsill in the afternoon sunlight",
    "A beautiful beach with palm trees and clear blue water",
    "A busy city street with people walking in the rain",
    "A cozy living room with a fireplace and bookshelves",
    "Fresh vegetables arranged on a wooden cutting board",
    "A vintage car parked on a quiet street",
    "Snow-capped mountains reflected in a still lake",
    "A beautiful garden with blooming flowers and butterflies",
    "A cup of coffee and a book on a table by the window",
    "A person hiking on a trail through a dense forest",
    "A plate of colorful macarons arranged in a pattern",
    "An old lighthouse standing on rocky coastal cliffs",
    "A dog playing fetch in a grassy field",
    "An aerial view of a winding river through a valley",
    "A street musician performing on a busy corner",
    "A peaceful meditation garden with a small pond",
    "A hot air balloon floating over a scenic landscape",
    "Trees, Travel and Tea!",
    "A refreshing beverage.",
    "A moment of indulgence.",
    "I love coffee and maybe 3 people",
    "The perfect thirst quencher.",
    "Your daily dose of delight.",
    "Taste the tradition.",
    "Savor the flavor.",
    "Refresh and rejuvenate.",
    "Unwind and enjoy.",
    "The taste of home.",
    "A treat for your senses.",
    "A taste of adventure.",
    "A moment of bliss.",
    "Your travel companion.",
    "Fuel for your journey.",
    "The essence of nature.",
    "The warmth of comfort.",
    "In riding a horse, we borrow freedom.",
    "No hour of life is wasted that is spent in the saddle.",
    "Winning is not a sometime thing; it's an all time thing",
    "We won tonight",
    "A sip of happiness.",
    "Pure indulgence.",
    "Quench your thirst, ignite your spirit.",
    "Awaken your senses, embrace the moment.",
    "The taste of faraway lands.",
    "A taste of home, wherever you are.",
    "Your daily dose of delight.",
    "Your moment of serenity.",
    "The perfect pick-me-up.",
    "The perfect way to unwind.",
    "Taste the difference.",
    "Experience the difference.",
    "A refreshing escape.",
    "A delightful escape.",
    "The taste of tradition, the spirit of adventure.",
    "The warmth of home, the joy of discovery.",
    "Your passport to flavor.",
    "Your ticket to tranquility.",
    "Sip, savor, and explore.",
    "Indulge, relax, and rejuvenate.",
    "The taste of wanderlust.",
    "The comfort of home.",
    "A journey for your taste buds.",
    "A haven for your senses.",
    "Your refreshing companion.",
    "Your delightful escape.",
    "Taste the world, one sip at a time.",
    "Embrace the moment, one cup at a time.",
    "The essence of exploration.",
    "The comfort of connection.",
    "Quench your thirst for adventure.",
    "Savor the moment of peace.",
    "The taste of discovery.",
    "The warmth of belonging.",
    "Your travel companion, your daily delight.",
    "Your moment of peace, your daily indulgence.",
    "The spirit of exploration, the comfort of home.",
    "The joy of discovery, the warmth of connection.",
    "Sip, savor, and set off on an adventure.",
    "Indulge, relax, and find your peace.",
    "A delightful beverage.",
    "A moment of relaxation.",
    "The perfect way to start your day.",
    "The perfect way to end your day.",
    "A treat for yourself.",
    "Something to savor.",
    "A moment of calm.",
    "A taste of something special.",
    "A refreshing pick-me-up.",
    "A comforting drink.",
    "A taste of adventure.",
    "A moment of peace.",
    "To ride on a horse is to fly without wings",
    "We are the champions!",
    "We love riding!",
    "Everywhere is a good road trip, just need to find the way",
    "When i have a camera in my hand, i know no fear.",
    "There are always two people in every picture: the photographer and the viewer.",
    "A small indulgence.",
    "A daily ritual.",
    "A way to connect with others.",
    "A way to connect with yourself.",
    "A taste of home.",
    "A taste of something new.",
    "A moment to enjoy.",
    "A moment to remember."
]


def load_model():
    """Load the sentence embedding model"""
    print("Loading sentence embedding model...")
    return SentenceTransformer('all-MiniLM-L6-v2')


def get_caption_recommendations(caption, sentence_model, top_n=5):
    """Find similar captions based on embedding similarity"""
    # Get embedding for the input caption
    caption_embedding = sentence_model.encode([caption])[0].reshape(1, -1)

    # Get embeddings for all sample captions
    sample_embeddings = sentence_model.encode(sample_captions)

    # Calculate similarity with all sample captions
    similarities = cosine_similarity(caption_embedding, sample_embeddings)[0]

    # Get top N similar captions
    top_indices = np.argsort(similarities)[::-1][:top_n]
    recommendations = [(sample_captions[i], float(similarities[i])) for i in top_indices]

    return recommendations


def main():
    parser = argparse.ArgumentParser(description="Get caption recommendations")
    parser.add_argument("--caption", type=str, required=True, help="Input caption to find recommendations for")
    parser.add_argument("--top", type=int, default=5, help="Number of recommendations to return")
    args = parser.parse_args()

    # Load model
    sentence_model = load_model()

    # Get recommendations
    recommendations = get_caption_recommendations(args.caption, sentence_model, args.top)

    # Print results
    print(f"\nInput Caption: {args.caption}\n")
    print("Recommended Similar Captions:")
    for idx, (caption, similarity) in enumerate(recommendations, 1):
        percentage = similarity * 100
        print(f"{idx}. {caption} ({percentage:.1f}% match)")


if __name__ == "__main__":
    main()
