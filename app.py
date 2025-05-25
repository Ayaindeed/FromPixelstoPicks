import os
from src.object_detector import ObjectDetector
from src.Hashtag_generator  import HashtagGenerator
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations
os.environ['PYTHONPATH'] = os.getcwd()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
import streamlit as st
import numpy as np
import torch
from PIL import Image
from src.recommendation_engine import sample_captions
from transformers import AutoTokenizer, VisionEncoderDecoderModel, ViTImageProcessor
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from src.evaluation import  ModelEvaluator

st.set_page_config(
    page_title="From Pixels to Picks",
    page_icon="🖼️",
    layout="wide"
)


def setup_sidebar():
    st.sidebar.markdown("""
        ### About ᶠʳᵒᵐ ᴾᶦˣᵉˡˢ ᵗᵒ ᴾᶦᶜᵏˢ
         Our app uses computer vision and NLP to analyze images:
         
        -Detects objects with YOLOv8
        
        -Generates captions via Vision Transformer
        
        -Finds similar captions using semantic analysis
        
        -Suggests hashtags for social sharing

        ---

        ### Settings
    """)

    settings = {
        'num_recommendations': st.sidebar.slider(
            "Number of Recommendations",
            min_value=1,
            max_value=10,
            value=5,
            help="Adjust the number of similar captions to display",
            key="slider_recommendations"
        ),
        'confidence_threshold': st.sidebar.slider(
            "Detection Confidence",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            help="Set the minimum confidence threshold for object detection",
            key="slider_confidence"
        ),
        'show_metrics': st.sidebar.checkbox(
            "Show Evaluation Metrics",
            value=False,
            help="Display detailed model performance metrics",
            key="checkbox_metrics"
        )
    }

    # Version info
    st.sidebar.markdown("---")
    st.sidebar.markdown("v2.0 • 2025")

    return settings


st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background-image: linear-gradient(to bottom, #ff6ec4, #7873f5);
        padding: 2rem;
    }
    
    .sidebar .sidebar-content {
        background: transparent;
    }
    
    /* Custom styling for sliders */
    .stSlider > div > div > div {
        background-color: rgba(255, 255, 255, 0.2);
    }
    
    /* Custom styling for checkboxes */
    .stCheckbox > div > div > div {
        background-color: rgba(255, 255, 255, 0.2);
        border-radius: 5px;
        padding: 2px;
    }
    
    /* Hover effects */
    .sidebar div:hover {
        transition: all 0.3s ease;
    }
    
    /* Tooltip styling */
    .stTooltip {
        color: white;
        background-color: rgba(0, 0, 0, 0.8);
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

os.makedirs("uploads", exist_ok=True)


@st.cache_resource
def load_models():
    """Load and cache models to avoid reloading on each run"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Image captioning model
    image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    caption_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning").to(device)

    # Sentence embedding model for similarity comparisons
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

    detector = ObjectDetector(conf_threshold=0.5)
    hashtag_generator = HashtagGenerator()
    evaluator = ModelEvaluator()

    return {
        'device': device,
        'image_processor': image_processor,
        'tokenizer': tokenizer,
        'caption_model': caption_model,
        'sentence_model': sentence_model,
        'object_detector': detector,
        'hashtag_generator': hashtag_generator,
        'evaluator': evaluator

    }



@st.cache_data
def get_sample_embeddings(_models):
    return _models['sentence_model'].encode(sample_captions)


def generate_caption(image, models):
    """Generate a caption for an image using the pre-trained model"""
    # Process image
    pixel_values = models['image_processor'](image, return_tensors="pt").pixel_values.to(models['device'])

    # Generate caption
    with torch.no_grad():
        output_ids = models['caption_model'].generate(
            pixel_values,
            max_length=50,
            num_beams=4,
            early_stopping=True
        )

    # Decode caption
    caption = models['tokenizer'].decode(output_ids[0], skip_special_tokens=True)
    return caption


# def get_caption_recommendations(caption, models, sample_embeddings, top_n=5):
#     """Find similar captions based on embedding similarity"""
#     # Get embedding for the generated caption
#     caption_embedding = models['sentence_model'].encode([caption])[0].reshape(1, -1)
#
#     # Calculate similarity with all sample captions
#     similarities = cosine_similarity(caption_embedding, sample_embeddings)[0]
#
#     # Get top N similar captions
#     top_indices = np.argsort(similarities)[::-1][:top_n]
#     recommendations = [(sample_captions[i], float(similarities[i])) for i in top_indices]
#
#     return recommendations

def get_enhanced_recommendations(caption, models, candidate_captions, top_n=5):
    """Enhanced version of caption recommendations with better matching"""

    caption_embedding = models['sentence_model'].encode([caption])[0].reshape(1, -1)
    candidate_embeddings = models['sentence_model'].encode(candidate_captions)
    similarities = cosine_similarity(caption_embedding, candidate_embeddings)[0]
    caption_terms = set(caption.lower().split())

    # Boost scores for captions that share key terms
    for i, candidate in enumerate(candidate_captions):
        candidate_terms = set(candidate.lower().split())
        term_overlap = len(caption_terms.intersection(candidate_terms))

        # Boost factor based on term overlap
        boost = 0.1 * term_overlap
        similarities[i] += boost

    # Get top N recommendations with boosted scores
    top_indices = np.argsort(similarities)[::-1][:top_n]
    recommendations = [(candidate_captions[i], float(similarities[i])) for i in top_indices]

    return recommendations


def main():
    settings = setup_sidebar()

    # Header
    st.markdown("""
            <h1 style='
                font-size: 48px;
                background: -webkit-linear-gradient(45deg, #ff6ec4, #7873f5);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-weight: bold;
                margin-bottom: 0.2em;
            '>From Pixels to Picks</h1>

            <p style='
                font-size: 20px;
                background: -webkit-linear-gradient(45deg, #f093fb, #f5576c);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-weight: 500;
                margin-top: 0;
            '>Just show me the image-I’ll do the rest (AYA trained me well ^-^).</p>
        """, unsafe_allow_html=True)
    num_recommendations = settings['num_recommendations']
    show_metrics = settings['show_metrics']
    confidence_threshold = settings['confidence_threshold']

    # Load models
    with st.spinner("Loading models..."):
        models = load_models()
        sample_embeddings = get_sample_embeddings(models)

    # Image upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # Process uploaded image
    if uploaded_file is not None:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Uploaded Image")
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, use_container_width=True)
        with col2:
            st.subheader("Object Detection")

            with st.spinner("Detecting objects (enhanced mode)..."):
                detections, annotated_img = models['object_detector'].detect_and_draw(image)
                detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)

                if annotated_img is not None:
                    st.image(annotated_img, use_container_width=True)

                st.subheader("Detected Objects")
                for det in detections:
                    confidence = det['confidence'] * 100
                    st.write(f"• {det['name']} ({confidence:.1f}%)")

        with col3:
            st.subheader("Image Description")
            with st.spinner("Generating caption..."):
                caption = generate_caption(image, models)
                st.success(caption)
                # Generate hashtags
                object_labels = [det['name'] for det in detections]
                hashtags = models['hashtag_generator'].generate_hashtags(caption, object_labels)

                # Generate hashtags using caption and detected objects
                with st.spinner("Generating hashtags..."):
                    # Get object labels from detections
                    object_labels = [det['name'] for det in detections]

                    # Generate hashtags
                    try:
                        hashtags = models['hashtag_generator'].generate_hashtags(
                            caption=caption,  # Named argument
                            object_labels=object_labels  # Named argument
                        )

                        # Display hashtags
                        if hashtags:
                            st.markdown("**Top Hashtags:**")
                            for tag in hashtags[:5]:
                                st.markdown(
                                    f"<span style='background-color:#e1bee7; color:#333; padding:4px 10px; border-radius:12px; margin:2px; display:inline-block;'>#{tag}</span>",
                                    unsafe_allow_html=True)
                        else:
                            st.write("No hashtags generated")
                    except Exception as e:
                        st.error(f"Error generating hashtags: {str(e)}")
                        st.write("Could not generate hashtags")

                # Display recommendations
                st.subheader("Recommended Captions")
                with st.spinner("Finding similar captions..."):
                    recommendations = get_enhanced_recommendations(
                        caption, models, sample_captions, num_recommendations
                    )

                    for i, (rec_caption, similarity) in enumerate(recommendations, 1):
                        percentage = similarity * 100
                        st.write(f"**{i}. {rec_caption}** - *{percentage:.1f}% match*")
            st.success("Results saved successfully!")

        # Option to save caption
        st.subheader("Save Generated Caption")
        if st.button("Save Caption to File"):
            with open("generated_captions.txt", "a") as f:
                f.write(f"Caption: {caption}\n")
                f.write(f"Image: {uploaded_file.name}\n")
                f.write("-" * 50 + "\n")
            st.success("Caption saved successfully!")

        #  evaluation metrics
        if show_metrics:
            st.subheader("Model Evaluation Metrics")
            eval_col1 = st.columns(1)[0]  # Access the first column of the tuple

            with eval_col1:
                st.markdown("##### Object Detection Metrics")
                # For demo p, we'll use the first detection as ground truth
                if len(detections) > 0:
                    ground_truth = detections[:1]  # Using first detection as ground truth
                    detection_metrics = models['evaluator'].evaluate_object_detection(
                        predictions=detections,
                        ground_truth=ground_truth
                    )

                    if detection_metrics:
                        st.write(f"Precision: {detection_metrics['precision']:.2f}")
                        st.write(f"Recall: {detection_metrics['recall']:.2f}")
                        st.write(f"F1-Score: {detection_metrics['f1_score']:.2f}")
                else:
                    st.write("No detections to evaluate")



if __name__ == "__main__":
    main()
