"""Service for making LLM API calls using LiteLLM."""
import os
from typing import List, Dict, Any, Optional
import logging
import litellm

from src.config.settings import app_settings


logger = logging.getLogger(__name__)


class LLMCallService:
    """A service wrapper around LiteLLM for making LLM API calls."""

    def __init__(self, 
                 model: Optional[str] = None,
                 api_key: Optional[str] = None,
                 api_base: Optional[str] = None,
                 temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None):
        """Initialize the LLM service with optional custom app_settings.
        Args:
            model: The LLM model to use
            api_key: The API key for the LLM provider
            api_base: The API base URL (required for some providers like Azure)
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens in the response
        """
        self.model = model or app_settings.model
        self.api_key = api_key or app_settings.api_key
        self.api_base = api_base or app_settings.api_base
        self.temperature = temperature or app_settings.temperature
        self.max_tokens = max_tokens or app_settings.max_tokens
        logging.getLogger("LiteLLM").disabled = True


    async def generate_completion(self, 
                                messages: List[Dict[str, Any]], 
                                **kwargs) -> Any:
        """Generate a completion using the configured LLM.
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional arguments to pass to litellm.completion
        Returns:
            The completion response from the LLM
        Raises:
            Exception: If the LLM call fails
        """
        try:
            # Merge instance app_settings with any provided kwargs
            call_kwargs = {
                "model": self.model,
                "api_key": self.api_key,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            
                
            # Override with any provided kwargs
            call_kwargs.update(kwargs)
            
            # Calculate token count before making the call
            self.report_token_use(messages)
            response = await litellm.acompletion(
                messages=messages,
                **call_kwargs
            )
            self.report_cost(response)
            return response
            
        except Exception as e:
            logger.error(f"Error in LLM call: {str(e)}")
            raise


    def report_cost(self, response: Any) -> None:
        """Get the cost of the LLM call from the response.
        Args:
            response: The response object from the LLM call
        Returns:
            The cost of the LLM call
        """
        cost = litellm.completion_cost(model=self.model, completion_response=response)
        formatted_string = f"${float(cost):.10f}"
        logger.info(f"LLM completion cost: {formatted_string}")
        

    def report_token_use(self, messages: List[Dict[str, Any]]) -> None:
        """Get the token count of the LLM call from the messages.
        Args:
            messages: The messages sent to the LLM
        Returns:
            The token count of the LLM call
        """
        token_count = litellm.token_counter(model=self.model, messages=messages)
        logger.info(f"LLM token count: {token_count}")
