import json
import requests
from typing import Any, Dict, Iterator, List, Mapping, Optional
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from langchain.llms.base import LLM

from services.auth import authenticate

class MyCustomLLM(LLM):
    """Custom LLM wrapper for a remote LLM API."""
    api_url: str 
    token: str
    model_id: str
    ID: str
    max_tokens: int = 4000  # Add default max_tokens

    @property
    def _llm_type(self) -> str:
        return "my_custom_llm"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """
        Call the LLM API with the provided prompt.
        
        Args:
            prompt: The input prompt text
            stop: Optional stop sequences
            
        Returns:
            str: The LLM response text
        """
        # Prepare headers
        headers = {
            'Accept': 'text/event-stream',
            'Content-Type': 'application/json',
            'Authorization': self.token 
        }
        
        # Prepare messages, e.g., system + user
        messages = [
            {
                "role": "system",
                "content": "You are an AI assistant that helps answering questions",
            },
            {
                "role": "user",
                "content": prompt,
            }
        ]

        # Prepare data payload with increased token limit
        data = {
            "id": self.ID,
            "modelId": self.model_id,
            "messages": messages,
            "maxTokens": self.max_tokens,  # Use instance variable
            "stream": False,
            "presencePenalty": 0,
            "stop": None,
            "frequencyPenalty": 0,
            "topP": 0.95,
            "temperature": 0.7,
            "safeGuardSettings": {
                "status": "monitor",
                "configurationStyle": "easy",
                "configuration": {
                    "moderation": "one",
                    "personalInformation": "one",
                    "promptInjection": "one",
                    "unknownLinks": "one"
                }
            },
            "locale": "en"
        }

        # Make the POST request
        try:
            response = requests.post(self.api_url, json=data, headers=headers, timeout=120)
            response.raise_for_status()
            result = response.json()
        except requests.exceptions.Timeout:
            return "Error: Request timed out. The document may be too large or complex."
        except requests.exceptions.RequestException as e:
            return f"Error: API request failed - {str(e)}"
        
        # Enhanced response parsing with better error messages
        try:
            # Try standard 'content' field first
            if "content" in result:
                content = result["content"]
                if content and content.strip():
                    return content
                else:
                    print("WARNING: 'content' field exists but is empty")
                    
            # Try alternative fields
            elif "message" in result:
                return result["message"]
            elif "text" in result:
                return result["text"]
            elif "output" in result:
                return result["output"]
                
            # If it's a nested structure, try common patterns
            elif "choices" in result and len(result["choices"]) > 0:
                # OpenAI-style format
                choice = result["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    return choice["message"]["content"]
                elif "text" in choice:
                    return choice["text"]
                    
            # Check for error messages in response
            elif "error" in result:
                error_msg = result["error"]
                if isinstance(error_msg, dict):
                    error_msg = error_msg.get("message", str(error_msg))
                print(f"API returned error: {error_msg}")
                return f"Error from API: {error_msg}"
                
            # Last resort: print structure for debugging and return error message
            else:
                print("ERROR: Unexpected API response structure:")
                print(json.dumps(result, indent=4))
                return f"Error: Could not extract content from API response. Keys available: {list(result.keys())}"
                
        except Exception as e:
            print(f"Error parsing API response: {e}")
            print("Full response:")
            print(json.dumps(result, indent=4))
            return f"Error parsing response: {str(e)}"