# Copyright(c) 2025 NVIDIA Corporation. All rights reserved.

# NVIDIA Corporation and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA Corporation is strictly prohibited.

import requests
import json
from loguru import logger
import asyncio
import os
from concurrent.futures import ThreadPoolExecutor

class GoogleTranslationService:
    """
    Translation service using Google Cloud Translation API with language auto-detection
    and dynamic target language handling.
    
    This implementation uses API Key authentication via direct REST API calls.
    """
    
    def __init__(self, project_id, default_target_language="en"):
        """
        Initialize Google Translation service
        
        Args:
            project_id: Google Cloud project ID
            default_target_language: Default target language code (ISO 639-1) used as fallback
        """
        # Get API key from environment variable
        self.api_key = os.environ.get("GOOGLE_TRANSLATE_API_KEY")
        
        # Log detailed API key status for better debugging
        if not self.api_key:
            logger.error("GOOGLE_TRANSLATE_API_KEY environment variable not set. Translation will fail.")
            logger.error("Please set the GOOGLE_TRANSLATE_API_KEY environment variable with a valid Google Cloud Translation API key.")
            # List available environment variables to help with debugging
            env_vars = [k for k in os.environ.keys() if k.startswith("GOOGLE_")]
            if env_vars:
                logger.info(f"Found these Google-related environment variables: {env_vars}")
            else:
                logger.info("No Google-related environment variables found.")
        else:
            masked_key = self.api_key[:4] + '*' * (len(self.api_key) - 8) + self.api_key[-4:] if len(self.api_key) > 8 else "****"
            logger.info(f"Found Google Translate API key: {masked_key}")
            
        self.project_id = project_id
        self.default_target_language = default_target_language
        
        # Define base URLs for the API
        self.translate_url = "https://translation.googleapis.com/language/translate/v2"
        self.detect_url = "https://translation.googleapis.com/language/translate/v2/detect"
        
        logger.info(f"Initialized Google Translation Service with default target language: {default_target_language}")
        
    async def translate_text(self, text, source_language=None, target_language=None):
        """
        Translate text with auto-detection of source language
        
        Args:
            text: Text to translate
            source_language: Source language code (if None, auto-detect)
            target_language: Target language code (defaults to self.default_target_language if None)
            
        Returns:
            Dictionary with translated text and detected language
        """
        if not text or text.strip() == "":
            return {"translated_text": "", "detected_language": None}
            
        if not self.api_key:
            logger.error("No API key available for translation")
            return {"translated_text": text, "detected_language": None}
            
        try:
            # Prepare the request data
            data = {
                "q": text,
                "target": target_language or self.default_target_language,
                "format": "text"
            }
            
            # Add source language if provided
            if source_language:
                data["source"] = source_language
                
            # Build the URL with the API key
            url = f"{self.translate_url}?key={self.api_key}"
            
            # Run the API request in a separate thread to avoid blocking
            loop = asyncio.get_running_loop()
            with ThreadPoolExecutor() as pool:
                response_data = await loop.run_in_executor(
                    pool,
                    lambda: self._make_api_request(url, data)
                )
            
            if response_data and "data" in response_data:
                translation = response_data["data"]["translations"][0]
                
                # Get detected language if source wasn't specified
                detected_language = None
                if source_language:
                    detected_language = source_language
                elif "detectedSourceLanguage" in translation:
                    detected_language = translation["detectedSourceLanguage"]
                
                return {
                    "translated_text": translation["translatedText"],
                    "detected_language": detected_language
                }
            else:
                logger.error(f"Translation failed: {response_data}")
                return {"translated_text": text, "detected_language": None}
                
        except Exception as e:
            logger.error(f"Translation error: {e}")
            # Provide more detailed error information
            if not self.api_key:
                logger.error("API key is missing. Set the GOOGLE_TRANSLATE_API_KEY environment variable.")
            return {"translated_text": text, "detected_language": None}
            
    async def detect_language(self, text):
        """
        Detect the language of the text
        
        Args:
            text: Text to detect language
            
        Returns:
            Detected language code
        """
        if not text or text.strip() == "":
            logger.debug(f"Empty text provided for language detection, returning None")
            return None
            
        # Special handling for very short text or special tokens
        if len(text.strip()) < 5:
            logger.debug(f"Text too short for reliable detection: [{text}]")
            
            # If it's something like (śmiech), we try a simple character-based heuristic
            if text.strip().startswith('(') and text.strip().endswith(')'):
                # Check for non-ASCII characters that might indicate non-English text
                has_non_ascii = any(ord(c) > 127 for c in text)
                if has_non_ascii:
                    # Very basic character set detection - for production, use a proper library
                    if 'ś' in text or 'ć' in text or 'ł' in text:
                        logger.debug(f"Detected Polish characters in special token: [{text}]")
                        return "pl"  # Polish
                    elif 'ñ' in text or 'á' in text or 'é' in text:
                        logger.debug(f"Detected Spanish characters in special token: [{text}]")
                        return "es"  # Spanish
                    elif 'ü' in text or 'ö' in text or 'ä' in text:
                        logger.debug(f"Detected German characters in special token: [{text}]")
                        return "de"  # German
                    # Add more as needed
            
        if not self.api_key:
            logger.error("No API key available for language detection")
            return None
            
        try:
            # Prepare the request data
            data = {
                "q": text
            }
            
            # Build the URL with the API key
            url = f"{self.detect_url}?key={self.api_key}"
            
            # Run the API request in a separate thread to avoid blocking
            loop = asyncio.get_running_loop()
            with ThreadPoolExecutor() as pool:
                response_data = await loop.run_in_executor(
                    pool,
                    lambda: self._make_api_request(url, data)
                )
            
            if response_data and "data" in response_data:
                detection = response_data["data"]["detections"][0][0]
                detected_lang = detection["language"]
                confidence = detection.get("confidence", 0)
                
                logger.debug(f"Detected language: {detected_lang} with confidence: {confidence:.2f} for text: [{text[:50]}{'...' if len(text) > 50 else ''}]")
                
                # If confidence is too low, handle carefully
                if confidence < 0.5:
                    logger.debug(f"Low confidence language detection ({confidence:.2f}). Result might be inaccurate.")
                    
                    # For very short text with low confidence, better to be cautious
                    if len(text.strip()) < 10:
                        logger.debug(f"Text too short with low confidence detection, consider using default")
                        # Here you could return None and let the caller decide,
                        # or implement more sophisticated fallback strategies
                
                return detected_lang
            else:
                logger.error(f"Language detection failed: {response_data}")
                return None
                
        except Exception as e:
            logger.error(f"Language detection error: {e}")
            return None
    
    def _make_api_request(self, url, data):
        """Make an API request to Google Translate API"""
        headers = {"Content-Type": "application/json"}
        
        try:
            logger.debug(f"Making API request to: {url.split('?')[0]} with data: {data}")
            response = requests.post(url, headers=headers, data=json.dumps(data))
            
            if response.status_code != 200:
                logger.error(f"API request failed with status code {response.status_code}")
                logger.error(f"Response: {response.text}")
                return None
                
            response.raise_for_status()  # Raise an exception for 4XX/5XX responses
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return None