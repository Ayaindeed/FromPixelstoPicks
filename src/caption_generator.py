import torch
from PIL import Image
from transformers import (
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer
)
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class CaptionGenerator:
    def __init__(self):
        """Initialize the caption generator with pre-trained src."""
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model_name = "nlpconnect/vit-gpt2-image-captioning"

            logger.info(f"Loading src from {self.model_name}...")
            self.image_processor = ViTImageProcessor.from_pretrained(self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name).to(self.device)

            logger.info("Caption generator initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing caption generator: {str(e)}")
            raise

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess the image for the model.

        Args:
            image: PIL Image object

        Returns:
            Tensor: Processed image tensor
        """
        try:
            return self.image_processor(
                image,
                return_tensors="pt"
            ).pixel_values.to(self.device)
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise

    def generate(
            self,
            image: Image.Image,
            num_captions: int = 5,
            max_length: int = 50,
            num_beams: int = 5,
            temperature: float = 1.0,
            early_stopping: bool = True
    ) -> List[str]:
        """
        Generate multiple captions for the given image.

        Args:
            image: PIL Image object
            num_captions: Number of captions to generate
            max_length: Maximum length of generated caption
            num_beams: Number of beams for beam search
            temperature: Temperature for sampling
            early_stopping: Whether to use early stopping

        Returns:
            List[str]: List of generated captions
        """
        try:
            # Preprocess image
            pixel_values = self.preprocess_image(image)

            # Generate multiple captions
            with torch.no_grad():
                output_ids = self.model.generate(
                    pixel_values,
                    max_length=max_length,
                    num_beams=num_beams,
                    num_return_sequences=num_captions,
                    temperature=temperature,
                    do_sample=True,
                    early_stopping=early_stopping,
                    no_repeat_ngram_size=2,
                    top_k=50,
                    top_p=0.95
                )

            # Decode captions
            captions = [
                self.tokenizer.decode(output, skip_special_tokens=True)
                for output in output_ids
            ]

            # Remove duplicates while preserving order
            unique_captions = []
            seen = set()
            for caption in captions:
                if caption not in seen:
                    unique_captions.append(caption)
                    seen.add(caption)

            logger.info(f"Generated {len(unique_captions)} diverse captions")
            return unique_captions

        except Exception as e:
            logger.error(f"Error generating captions: {str(e)}")
            return ["Error generating caption"]


    def get_model_info(self) -> Dict:
        """
        Get information about the loaded src.

        Returns:
            Dict containing model information
        """
        return {
            'model_name': self.model_name,
            'device': str(self.device),
            'model_type': self.model.__class__.__name__,
            'tokenizer_type': self.tokenizer.__class__.__name__,
            'processor_type': self.image_processor.__class__.__name__
        }

    def __call__(self, image: Image.Image, **kwargs) -> str:
        """Allow using the class instance as a callable."""
        return self.generate(image, **kwargs)

    def cleanup(self):
        """Clean up resources if needed."""
        try:
            if hasattr(self, 'model'):
                del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Cleaned up caption generator resources")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")