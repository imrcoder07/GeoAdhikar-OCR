"""
Core OCR Processor using Google Gemini API
"""
import os
import base64
import json
import time
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import google.generativeai as genai
from PIL import Image
import cv2
import numpy as np
from loguru import logger
from config.config import Config

class GeminiOCRProcessor:
    """Main OCR processor using Google Gemini API"""
    
    def __init__(self):
        """Initialize the Gemini OCR processor"""
        self.api_key = Config.GEMINI_API_KEY
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Initialize retry settings
        self.max_retries = Config.MAX_RETRIES
        self.retry_delay = 2  # seconds
        
        logger.info("Gemini OCR Processor initialized successfully")
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess image for better OCR results
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image array
        """
        # Read image
        img = cv2.imread(image_path)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
        
        # Deskew if needed
        angle = self._get_skew_angle(denoised)
        if abs(angle) > 0.5:
            denoised = self._rotate_image(denoised, angle)
        
        return denoised
    
    def _get_skew_angle(self, image: np.ndarray) -> float:
        """Detect skew angle of the image"""
        coords = np.column_stack(np.where(image > 0))
        if len(coords) > 0:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            return angle
        return 0
    
    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by given angle"""
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        return rotated
    
    def extract_text_from_image(self, image_path: str, language_hints: List[str] = None) -> Dict:
        """
        Extract text from image using Gemini API
        
        Args:
            image_path: Path to the image file
            language_hints: List of language codes (e.g., ['hi', 'en'])
            
        Returns:
            Dictionary containing OCR results
        """
        try:
            # Preprocess image
            logger.info(f"Processing image: {image_path}")
            preprocessed = self.preprocess_image(image_path)
            
            # Save preprocessed image temporarily
            temp_path = Path(image_path).parent / f"temp_{Path(image_path).name}"
            cv2.imwrite(str(temp_path), preprocessed)
            
            # Open image with PIL
            img = Image.open(temp_path)
            
            # Prepare language hints
            if not language_hints:
                language_hints = Config.DEFAULT_LANGUAGE_HINTS
            
            # Create prompt for Gemini
            prompt = self._create_ocr_prompt(language_hints)
            
            # Call Gemini API with retries
            response = self._call_gemini_with_retry(img, prompt)
            
            # Clean up temp file
            if temp_path.exists():
                os.remove(temp_path)
            
            # Parse response
            result = self._parse_gemini_response(response)
            
            logger.info(f"Successfully extracted text from {image_path}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            raise
    
    def _create_ocr_prompt(self, language_hints: List[str]) -> str:
        """Create prompt for Gemini to extract FRA document text"""
        languages = ", ".join(language_hints)
        
        prompt = f"""
        You are an expert OCR system for Indian government Forest Rights Act (FRA) documents.
        
        Extract ALL text from this document image. The document may contain text in: {languages}.
        
        Please extract and structure the following information if present:
        1. Patta ID/Number
        2. Holder Name (person's name)
        3. Father's/Relative's Name
        4. Village Name
        5. District
        6. State
        7. Land Area (with units)
        8. Issue Date
        9. Claim Type (IFR/CR/CFR)
        10. Status (Granted/Pending/Rejected)
        11. Any coordinates or survey numbers
        
        IMPORTANT:
        - Extract ALL visible text, even if partially visible or unclear
        - Preserve original text alongside any transliterations
        - For Hindi/regional text, provide both original and romanized versions
        - Include confidence indicators for unclear text
        - Note any stamps, signatures, or official marks
        
        Return the result in this JSON format:
        {{
            "raw_text": "complete raw text from document",
            "structured_fields": {{
                "patta_id": {{"value": "", "raw": "", "confidence": 0.0}},
                "holder_name": {{"value": "", "raw": "", "confidence": 0.0}},
                "relative_name": {{"value": "", "raw": "", "confidence": 0.0}},
                "village": {{"value": "", "raw": "", "confidence": 0.0}},
                "district": {{"value": "", "raw": "", "confidence": 0.0}},
                "state": {{"value": "", "raw": "", "confidence": 0.0}},
                "area": {{"value": "", "unit": "", "raw": "", "confidence": 0.0}},
                "issue_date": {{"value": "", "raw": "", "confidence": 0.0}},
                "claim_type": {{"value": "", "raw": "", "confidence": 0.0}},
                "status": {{"value": "", "raw": "", "confidence": 0.0}}
            }},
            "language_detected": [],
            "quality_score": 0.0,
            "has_stamp": false,
            "has_signature": false
        }}
        
        Extract text now:
        """
        
        return prompt
    
    def _call_gemini_with_retry(self, image: Image.Image, prompt: str) -> str:
        """Call Gemini API with retry logic"""
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Calling Gemini API (attempt {attempt + 1}/{self.max_retries})")
                
                # Generate content
                response = self.model.generate_content([prompt, image])
                
                # Check if response is valid
                if response and response.text:
                    return response.text
                else:
                    raise ValueError("Empty response from Gemini")
                    
            except Exception as e:
                logger.warning(f"Gemini API call failed (attempt {attempt + 1}): {str(e)}")
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise Exception(f"Failed to call Gemini after {self.max_retries} attempts: {str(e)}")
    
    def _parse_gemini_response(self, response_text: str) -> Dict:
        """Parse Gemini response and extract structured data"""
        try:
            # Try to extract JSON from response
            # Sometimes Gemini adds markdown formatting
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_str = response_text[json_start:json_end].strip()
            elif "{" in response_text and "}" in response_text:
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                json_str = response_text[json_start:json_end]
            else:
                # If no JSON found, create basic structure
                return {
                    "raw_text": response_text,
                    "structured_fields": {},
                    "language_detected": ["unknown"],
                    "quality_score": 0.5,
                    "parse_error": "Could not extract JSON from response"
                }
            
            # Parse JSON
            data = json.loads(json_str)
            
            # Ensure all required fields are present
            if "structured_fields" not in data:
                data["structured_fields"] = {}
            
            if "raw_text" not in data:
                data["raw_text"] = response_text
            
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from Gemini response: {str(e)}")
            return {
                "raw_text": response_text,
                "structured_fields": {},
                "language_detected": ["unknown"],
                "quality_score": 0.5,
                "parse_error": str(e)
            }
    
    def process_batch(self, image_paths: List[str], language_hints: List[str] = None) -> List[Dict]:
        """
        Process multiple images in batch
        
        Args:
            image_paths: List of image file paths
            language_hints: Language codes
            
        Returns:
            List of OCR results
        """
        results = []
        total = len(image_paths)
        
        for idx, image_path in enumerate(image_paths, 1):
            logger.info(f"Processing image {idx}/{total}: {image_path}")
            
            try:
                result = self.extract_text_from_image(image_path, language_hints)
                result["file_path"] = image_path
                result["status"] = "success"
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {str(e)}")
                results.append({
                    "file_path": image_path,
                    "status": "error",
                    "error": str(e)
                })
            
            # Add delay between requests to avoid rate limiting
            if idx < total:
                time.sleep(1)
        
        return results
    
    def extract_from_pdf(self, pdf_path: str, language_hints: List[str] = None) -> List[Dict]:
        """
        Extract text from PDF file
        
        Args:
            pdf_path: Path to PDF file
            language_hints: Language codes
            
        Returns:
            List of OCR results for each page
        """
        from pdf2image import convert_from_path
        
        logger.info(f"Converting PDF to images: {pdf_path}")
        
        # Convert PDF to images
        images = convert_from_path(pdf_path, dpi=300)
        
        results = []
        for idx, image in enumerate(images, 1):
            logger.info(f"Processing page {idx}/{len(images)}")
            
            # Save image temporarily
            temp_image_path = f"temp_page_{idx}.png"
            image.save(temp_image_path)
            
            try:
                # Process image
                result = self.extract_text_from_image(temp_image_path, language_hints)
                result["page_number"] = idx
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to process page {idx}: {str(e)}")
                results.append({
                    "page_number": idx,
                    "status": "error",
                    "error": str(e)
                })
            
            finally:
                # Clean up temp file
                if os.path.exists(temp_image_path):
                    os.remove(temp_image_path)
        
        return results
