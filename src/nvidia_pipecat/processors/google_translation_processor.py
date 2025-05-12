# Copyright(c) 2025 NVIDIA Corporation. All rights reserved.

# NVIDIA Corporation and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA Corporation is strictly prohibited.

from loguru import logger
from pipecat.frames.frames import TranscriptionFrame, TextFrame, StopInterruptionFrame, UserStoppedSpeakingFrame, TTSSpeakFrame, LLMFullResponseStartFrame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
import time

# Shared language detection state across processor instances
class SharedLanguageState:
    detected_language = None

class GoogleSourceToEnglishTranslationProcessor(FrameProcessor):
    """
    Processor that translates text from any source language to English.
    
    This processor works with frames that have either:
    1. A "text" attribute (like TranscriptionFrame)
    2. A "transcript" attribute (like UserStoppedSpeakingTranscriptFrame)
    3. A "text" field in their metadata
    
    It detects the language and translates if needed, preserving the original
    text and detected language information in the frame's metadata.
    """
    
    DEFAULT_SPECIAL_TOKEN_THRESHOLD = 2
    DEFAULT_SPECIAL_TOKEN_INJECTION_TEXT = "Hello, are you there? Can you help me?"

    def __init__(self, 
                 translation_service, 
                 special_token_threshold=None,
                 special_token_injection_text=None,
                 **kwargs):
        """
        Initialize the processor with a translation service.
        
        Args:
            translation_service: The translation service to use for language detection and translation
            special_token_threshold (int, optional): Number of consecutive special tokens 
                                                     before injecting a message. 
                                                     Defaults to DEFAULT_SPECIAL_TOKEN_THRESHOLD.
            special_token_injection_text (str, optional): Text to inject after threshold is met.
                                                          Defaults to DEFAULT_SPECIAL_TOKEN_INJECTION_TEXT.
            **kwargs: Additional arguments passed to parent FrameProcessor.
        """
        super().__init__(**kwargs)
        self.translation_service = translation_service
        self.consecutive_special_tokens = 0
        self.special_token_threshold = special_token_threshold if special_token_threshold is not None else self.DEFAULT_SPECIAL_TOKEN_THRESHOLD
        self.special_token_injection_text = special_token_injection_text if special_token_injection_text is not None else self.DEFAULT_SPECIAL_TOKEN_INJECTION_TEXT
        self._language_history = []
        self._current_language = None
        self._logger = logger
    
    async def _translate_to_english(self, text):
        """
        Translate text from any language to English.
        
        Args:
            text: Text to translate
            
        Returns:
            Dictionary with translation results. Includes an "error" key on failure.
        """
        try:
            result = await self.translation_service.translate_text(
                text, 
                source_language=None,  # Auto detect source language
                target_language="en"
            )
            return result
        except Exception as e:
            logger.error(f"Error during source to English translation for text '{text[:50]}...': {e}")
            # Return original text as if it were English, with error information
            return {
                "translated_text": text,
                "detected_language": "en", # Fallback, assuming it might be English or pass-through
                "original_text": text,
                "error": str(e)
            }
        
    async def _inject_message_for_special_tokens(self):
        """
        Create and inject a message to prompt LLM response after consecutive special tokens.
        """
        injection_text = self.special_token_injection_text
        logger.info(f"Injected message to prompt LLM: '{injection_text}'")
        
        # Create a new frame with the injection text including required parameters
        # Add a default user_id and current timestamp
        frame = TranscriptionFrame(text=injection_text, user_id="system", timestamp=time.time())
        frame.metadata = {
            "text": injection_text,
            "detected_language": "en",
            "is_injected_message": True
        }
        
        # Push the frame downstream
        await self.push_frame(frame, FrameDirection.DOWNSTREAM)
    
    async def process_frame(self, frame, direction):
        """
        Process a frame, translating any text content from the source language to English.
        
        Args:
            frame: The frame to process
            direction: The direction of the frame
        """
        # Always call super().process_frame() first for proper initialization
        await super().process_frame(frame, direction)
        
        frame_type = type(frame).__name__
        logger.debug(f"Processing frame of type: {frame_type}")
        
        text_to_translate = None
        is_special_token = False
        
        # Extract text to translate from various frame types
        if hasattr(frame, "text") and isinstance(getattr(frame, "text", None), str):
            text_to_translate = frame.text
            logger.debug(f"Found frame with text: [{text_to_translate}]")
            
            # Check if it's a special token (sound effect, etc.)
            if text_to_translate.startswith("(") and text_to_translate.endswith(")"):
                is_special_token = True
                logger.debug(f"Text appears to be a special token/emotion: {text_to_translate}")
        elif hasattr(frame, "transcript") and isinstance(frame.transcript, str):
            text_to_translate = frame.transcript
            logger.debug(f"Processing frame with transcript: [{text_to_translate}]")
            
            # Check if it's a special token (sound effect, etc.)
            if text_to_translate.startswith("(") and text_to_translate.endswith(")"):
                is_special_token = True
                logger.debug(f"Text appears to be a special token/emotion: {text_to_translate}")
                
        # If we found text to translate and it's a special token
        if text_to_translate and is_special_token:
            # Special token handling
            logger.debug(f"Special token detected, passing through without translation: {text_to_translate}")
            
            # Increment our counter for consecutive special tokens
            self.consecutive_special_tokens += 1
            logger.debug(f"Consecutive special tokens: {self.consecutive_special_tokens}")
            
            # Pass through the original frame without modification
            await self.push_frame(frame, direction)
            
            # If we've reached our threshold of consecutive special tokens, inject a message
            if self.consecutive_special_tokens >= self.special_token_threshold:
                logger.info(f"Detected {self.consecutive_special_tokens} consecutive special tokens. Injecting a message to prompt LLM response.")
                
                # Reset the counter
                self.consecutive_special_tokens = 0
                
                # Inject a message to prompt LLM response
                await self._inject_message_for_special_tokens()
            return
            
        # For normal text, reset the special token counter
        if text_to_translate and not is_special_token:
            self.consecutive_special_tokens = 0
            
            # Only translate if there's text to translate
            if text_to_translate.strip() != "":
                # Translate the text to English
                result = await self._translate_to_english(text_to_translate)
                
                # Ensure metadata attribute exists
                if not hasattr(frame, "metadata"):
                    frame.metadata = {}
                
                # Store the original text
                frame.metadata["original_text"] = text_to_translate
                
                if "error" in result:
                    logger.warning(f"Translation to English failed for '{text_to_translate[:50]}...'. Passing through original text. Error: {result['error']}")
                    # Use original text as translated_text and 'en' as detected_language
                    translated_text = text_to_translate
                    detected_language = "en" # Fallback
                    frame.metadata["translation_error"] = result["error"]
                else:
                    translated_text = result["translated_text"]
                    detected_language = result["detected_language"]
                
                # Store the detected language in metadata for downstream processors
                if detected_language: # Can be None if auto-detect fails and no error was caught
                    frame.metadata["conversation_language"] = detected_language
                    logger.debug(f"Setting conversation_language={detected_language} from translation result")
                    # Store the detected language in shared state
                    SharedLanguageState.detected_language = detected_language
                    logger.debug(f"Updated shared language state to: {SharedLanguageState.detected_language}")
                
                # Store translated text and language information in metadata
                frame.metadata["text"] = translated_text
                frame.metadata["detected_language"] = detected_language # This might be the fallback 'en' on error
                
                if "error" not in result:
                    logger.info(f"Translated from {detected_language} to en: '{text_to_translate}' -> '{translated_text}'")
                
                # Update frame text fields
                if hasattr(frame, "text"):
                    frame.text = translated_text
                    
                if hasattr(frame, "transcript"):
                    frame.transcript = translated_text
        
        # Always forward the frame, whether modified or not
        await self.push_frame(frame, direction)

    def _is_likely_noise(self, text):
        """Identify if text is likely just background noise."""
        if not text or len(text.strip()) < 3:
            return True
            
        # Check for bracketed content with noise markers
        if '[' in text and ']' in text:
            noise_markers = ['...', 'music', 'background', 'plays', 'silence', 'noise']
            return any(marker in text.lower() for marker in noise_markers)
            
        return False
    
    def _update_language_history(self, language):
        """Update language history and determine the most consistent language."""
        if not language:
            return
            
        # Add to history and keep last 5 detections
        self._language_history.append(language)
        if len(self._language_history) > 5:
            self._language_history.pop(0)
            
        # Count occurrences of each language
        counts = {}
        for lang in self._language_history:
            counts[lang] = counts.get(lang, 0) + 1
            
        # Find the most frequent language (with at least 2 occurrences)
        primary_language = None
        max_count = 1
        for lang, count in counts.items():
            if count > max_count:
                max_count = count
                primary_language = lang
                
        # Update current language if we have a consistent detection
        if primary_language:
            self._current_language = primary_language
        elif not self._current_language:
            self._current_language = language


class GoogleEnglishToTargetTranslationProcessor(FrameProcessor):
    """
    Processor that translates text from English to the target language.
    
    This processor uses the conversation_language set by the
    GoogleSourceToEnglishTranslationProcessor to determine the target language.
    It buffers text from LLM responses (between LLMFullResponseStartFrame and
    LLMFullResponseEndFrame) to translate complete sentences for better quality.
    """
    
    def __init__(self, translation_service, default_target_language="en", **kwargs):
        """
        Initialize the processor with a translation service.
        
        Args:
            translation_service: The translation service to use for translation
            default_target_language: Default target language if none can be determined
            **kwargs: Additional arguments passed to parent FrameProcessor.
        """
        super().__init__(**kwargs)
        self.translation_service = translation_service
        self.default_target_language = default_target_language
        self.current_conversation_language = default_target_language
        # For buffering LLM response text
        self._llm_response_buffer = []
        self._is_processing_llm_response = False
        self._logger = logger
    
    async def _translate_to_target(self, text, target_language):
        """
        Translate text from English to the target language.
        
        Args:
            text: Text to translate
            target_language: Target language code
            
        Returns:
            Dictionary with translation results. Includes an "error" key on failure.
        """
        try:
            result = await self.translation_service.translate_text(
                text, 
                source_language="en",
                target_language=target_language
            )
            return result
        except Exception as e:
            logger.error(f"Error during English to {target_language} translation for text '{text[:50]}...': {e}")
            # Return original English text with error information
            return {
                "translated_text": text, # Fallback to original English text
                "error": str(e)
            }
    
    async def process_frame(self, frame, direction):
        """
        Process a frame, translating any text content from English to the target language.
        
        Args:
            frame: The frame to process
            direction: The direction of the frame
        """
        # Always call super().process_frame() first for proper initialization
        await super().process_frame(frame, direction)
        
        frame_type = type(frame).__name__
        logger.debug(f"GoogleEnglishToTargetTranslation processing frame of type: {frame_type}")
        
        # Update the current conversation language if present in metadata
        if hasattr(frame, "metadata") and "conversation_language" in frame.metadata:
            lang = frame.metadata["conversation_language"]
            if lang and lang != self.current_conversation_language:
                self.current_conversation_language = lang
                logger.debug(f"Updated conversation language from frame: {self.current_conversation_language}")
        # Check the shared language state for updates
        elif SharedLanguageState.detected_language and SharedLanguageState.detected_language != self.current_conversation_language:
            self.current_conversation_language = SharedLanguageState.detected_language
            logger.debug(f"Updated conversation language from shared state: {self.current_conversation_language}")

        # Handle LLM response stream for aggregation
        if frame_type == "LLMFullResponseStartFrame":
            logger.debug("LLMFullResponseStartFrame detected. Starting to buffer LLM response for translation.")
            self._is_processing_llm_response = True
            self._llm_response_buffer = []
            await self.push_frame(frame, direction) # Forward the original frame
            return

        if frame_type == "TextFrame" and self._is_processing_llm_response:
            text_chunk = None
            if hasattr(frame, "text") and isinstance(frame.text, str):
                text_chunk = frame.text
            elif hasattr(frame, "metadata") and "text" in frame.metadata and isinstance(frame.metadata["text"], str):
                # Fallback if text is in metadata, though typical for TextFrame is .text
                text_chunk = frame.metadata["text"]
            
            if text_chunk:
                self._llm_response_buffer.append(text_chunk)
                logger.debug(f"Buffered LLM text chunk: '{text_chunk}'")
            else:
                logger.debug(f"TextFrame in LLM stream has no text. Skipping add to buffer.")
            # Do not forward these individual English TextFrames; they are consumed by the buffer.
            return

        if frame_type == "LLMFullResponseEndFrame":
            if self._is_processing_llm_response:
                logger.debug("LLMFullResponseEndFrame detected. Processing buffered LLM response.")
                self._is_processing_llm_response = False
                
                full_english_text = "".join(self._llm_response_buffer)
                self._llm_response_buffer = [] # Clear buffer

                final_text_to_push = full_english_text
                
                if full_english_text.strip():
                    # Check shared state again before translation
                    if SharedLanguageState.detected_language and SharedLanguageState.detected_language != self.current_conversation_language:
                        self.current_conversation_language = SharedLanguageState.detected_language
                        logger.debug(f"Updated target language from shared state before translation: {self.current_conversation_language}")
                    
                    target_language = self.current_conversation_language
                    logger.debug(f"Aggregated English text for translation: '{full_english_text}' to target language: {target_language}")

                    if target_language != "en":
                        logger.info(f"TRANSLATING full aggregated response from English to {target_language}")
                        result = await self._translate_to_target(full_english_text, target_language)
                        
                        if "error" in result:
                            logger.warning(f"Full sentence translation to {target_language} failed for '{full_english_text[:50]}...'. Error: {result['error']}. Sending original English text.")
                            # final_text_to_push is already full_english_text
                        else:
                            final_text_to_push = result["translated_text"]
                            logger.info(f"Full sentence translated from en to {target_language}: '{full_english_text}' -> '{final_text_to_push}'")
                    else:
                        logger.debug("Target language is English, sending aggregated English text as is.")
                        # final_text_to_push is already full_english_text
                    
                    # Create a new TextFrame with the full (translated or original if error/English) sentence
                    # This new frame will go to TTS or other downstream services.
                    processed_text_frame = TextFrame(text=final_text_to_push)
                    # Attempt to carry over metadata from the EndFrame, or create new
                    if hasattr(frame, "metadata") and frame.metadata is not None:
                        processed_text_frame.metadata = frame.metadata.copy() 
                    else:
                        processed_text_frame.metadata = {}
                    
                    processed_text_frame.metadata["english_full_sentence"] = full_english_text 
                    processed_text_frame.metadata["is_full_translation_attempted"] = True
                    await self.push_frame(processed_text_frame, direction)
                
                else:
                    logger.debug("Buffered LLM response was empty or whitespace. No translation needed.")

                await self.push_frame(frame, direction) # Forward the original LLMFullResponseEndFrame
                return
            else: 
                logger.warning("LLMFullResponseEndFrame received without an active LLM response buffering session. Forwarding as is.")
                await self.push_frame(frame, direction)
                return

        # Fallback for frames not handled by the LLM buffering logic 
        # (e.g., standalone TextFrames or TTSSpeakFrames for fillers not in an LLM stream)
        text_to_translate = None
        is_special_token = False
        
        # Extract text to translate
        if hasattr(frame, "text") and isinstance(getattr(frame, "text", None), str):
            text_to_translate = frame.text
            
            # Check if it's a special token (sound effect, etc.)
            if text_to_translate.startswith("(") and text_to_translate.endswith(")"):
                is_special_token = True
                logger.debug(f"Special token detected, passing through without translation: {text_to_translate}")
                await self.push_frame(frame, direction)
                return
                
        elif hasattr(frame, "metadata") and "text" in frame.metadata:
            text_to_translate = frame.metadata["text"]
            
            # Check if it's a special token (sound effect, etc.)
            if text_to_translate.startswith("(") and text_to_translate.endswith(")"):
                is_special_token = True
                logger.debug(f"Special token detected, passing through without translation: {text_to_translate}")
                await self.push_frame(frame, direction)
                return
        
        # Check for standalone frame like TTSSpeakFrame - check shared state
        if frame_type in ["TTSSpeakFrame"]:
            if SharedLanguageState.detected_language and SharedLanguageState.detected_language != self.current_conversation_language:
                self.current_conversation_language = SharedLanguageState.detected_language
                logger.debug(f"For standalone frame {frame_type}, updated language from shared state: {self.current_conversation_language}")
        
        # Use the persistently stored conversation language as the target
        target_language = self.current_conversation_language
        logger.debug(f"Using target language: {target_language} for frame type {frame_type}")
        logger.debug(f"Current conversation language: {self.current_conversation_language}")

        # Skip if no text or empty text
        if not text_to_translate or text_to_translate.strip() == "":
            await self.push_frame(frame, direction)
            return
            
        # Ensure frame has metadata
        if not hasattr(frame, "metadata"):
            frame.metadata = {}
            
        # Store the English text for reference
        frame.metadata["english_text"] = text_to_translate
        
        # Check if this is a response to an injected message from special token handling
        if "is_injected_message" in frame.metadata: # Simplified check
            logger.info(f"Detected response to an injected message from special token handling")
        
        # Only translate if the target language is not English
        if target_language != "en":
            # This is where TTSSpeakFrame for a filler (e.g. "Let me look it up for you") would be translated.
            logger.info(f"TRANSLATING response from English to {target_language} for frame type {frame_type}")
            result = await self._translate_to_target(text_to_translate, target_language)
            
            translated_text_for_frame = text_to_translate # Default to original English if error

            if "error" in result:
                logger.warning(f"Translation to {target_language} failed for '{text_to_translate[:50]}...'. Passing through English text. Error: {result['error']}")
                frame.metadata["translation_error"] = result["error"]
                # translated_text_for_frame is already text_to_translate (English)
            else:
                translated_text_for_frame = result["translated_text"]
                logger.info(f"Translated from en to {target_language}: '{text_to_translate}' -> '{translated_text_for_frame}'")

            # Update frame text fields
            if hasattr(frame, "text"):
                frame.text = translated_text_for_frame
            if "text" in frame.metadata: # Check if 'text' key exists before assigning
                frame.metadata["text"] = translated_text_for_frame
            
        else:
            logger.debug(f"Target language is English, skipping translation for frame type {frame_type}")
            # Pass through frame as is if target is English (text is already in English)
            if hasattr(frame, "text"): # Ensure frame.text matches text_to_translate if it was from metadata
                frame.text = text_to_translate
            if "text" in frame.metadata:
                 frame.metadata["text"] = text_to_translate
        
        # Always forward the frame
        await self.push_frame(frame, direction)