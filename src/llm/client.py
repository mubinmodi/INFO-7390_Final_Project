"""
Unified LLM Client supporting OpenAI and Google Gemini
"""
import logging
from typing import Optional, List, Dict
from config.settings import (
    LLM_PROVIDER,
    OPENAI_API_KEY, OPENAI_MODEL,
    GEMINI_API_KEY, GEMINI_MODEL,
    TEMPERATURE, MAX_RETRIES
)

logger = logging.getLogger(__name__)


class LLMClient:
    """Unified interface for OpenAI and Gemini"""
    
    def __init__(self, 
                 provider: Optional[str] = None,
                 model: Optional[str] = None,
                 temperature: float = TEMPERATURE):
        """
        Initialize LLM client
        
        Args:
            provider: "openai" or "gemini" (defaults to LLM_PROVIDER from settings)
            model: Model name (defaults to provider's default model)
            temperature: Sampling temperature
        """
        self.provider = provider or LLM_PROVIDER
        self.temperature = temperature
        
        if self.provider == "openai":
            self.model = model or OPENAI_MODEL
            self._init_openai()
        elif self.provider == "gemini":
            self.model = model or GEMINI_MODEL
            self._init_gemini()
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    def _init_openai(self):
        """Initialize OpenAI client"""
        try:
            from openai import OpenAI
            
            if not OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY not set")
            
            self.client = OpenAI(api_key=OPENAI_API_KEY)
            logger.info(f"Initialized OpenAI: {self.model}")
            
        except Exception as e:
            logger.error(f"Error initializing OpenAI: {e}")
            raise
    
    def _init_gemini(self):
        """Initialize Gemini client"""
        try:
            import google.generativeai as genai
            
            if not GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY not set")
            
            genai.configure(api_key=GEMINI_API_KEY)
            self.client = genai.GenerativeModel(self.model)
            logger.info(f"Initialized Gemini: {self.model}")
            
        except Exception as e:
            logger.error(f"Error initializing Gemini: {e}")
            raise
    
    def generate(self, 
                prompt: str,
                system_prompt: Optional[str] = None,
                max_tokens: Optional[int] = None) -> str:
        """
        Generate text completion
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        for attempt in range(MAX_RETRIES):
            try:
                if self.provider == "openai":
                    return self._generate_openai(prompt, system_prompt, max_tokens)
                else:
                    return self._generate_gemini(prompt, system_prompt, max_tokens)
                    
            except Exception as e:
                logger.warning(f"Generation attempt {attempt + 1} failed: {e}")
                if attempt == MAX_RETRIES - 1:
                    raise
        
        raise Exception("Failed to generate after max retries")
    
    def _generate_openai(self, 
                        prompt: str, 
                        system_prompt: Optional[str],
                        max_tokens: Optional[int]) -> str:
        """Generate using OpenAI"""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
    
    def _generate_gemini(self,
                        prompt: str,
                        system_prompt: Optional[str],
                        max_tokens: Optional[int]) -> str:
        """Generate using Gemini"""
        # Combine system and user prompts
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        generation_config = {
            "temperature": self.temperature,
        }
        
        if max_tokens:
            generation_config["max_output_tokens"] = max_tokens
        
        response = self.client.generate_content(
            full_prompt,
            generation_config=generation_config
        )
        
        return response.text
    
    def chat(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None) -> str:
        """
        Chat completion with message history
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response
        """
        if self.provider == "openai":
            return self._chat_openai(messages, max_tokens)
        else:
            return self._chat_gemini(messages, max_tokens)
    
    def _chat_openai(self, messages: List[Dict[str, str]], max_tokens: Optional[int]) -> str:
        """Chat using OpenAI"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
    
    def _chat_gemini(self, messages: List[Dict[str, str]], max_tokens: Optional[int]) -> str:
        """Chat using Gemini"""
        # Convert messages to Gemini format
        # Combine system messages and convert roles
        prompt_parts = []
        
        for msg in messages:
            role = msg['role']
            content = msg['content']
            
            if role == 'system':
                prompt_parts.append(f"System: {content}")
            elif role == 'user':
                prompt_parts.append(f"User: {content}")
            elif role == 'assistant':
                prompt_parts.append(f"Assistant: {content}")
        
        full_prompt = "\n\n".join(prompt_parts)
        
        generation_config = {
            "temperature": self.temperature,
        }
        
        if max_tokens:
            generation_config["max_output_tokens"] = max_tokens
        
        response = self.client.generate_content(
            full_prompt,
            generation_config=generation_config
        )
        
        return response.text


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Test with Gemini
    client = LLMClient(provider="gemini")
    
    response = client.generate(
        prompt="What are the key financial metrics to analyze for a tech company?",
        system_prompt="You are a financial analyst expert."
    )
    
    print(f"Provider: {client.provider}")
    print(f"Model: {client.model}")
    print(f"Response: {response[:200]}...")