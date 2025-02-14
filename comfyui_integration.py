import os
import sys
import logging
import time
import requests
import json
from PIL import Image
from io import BytesIO
import random
import re
import textwrap

class ComfyUIIntegrator:
    def __init__(self, server_url="http://127.0.0.1:8188"):
        """
        Initialize ComfyUI Integration
        
        Args:
            server_url (str): URL of the ComfyUI server
        """
        # Add ComfyUI directory to Python path
        comfyui_path = os.path.join(os.path.dirname(__file__), 'ComfyUI')
        if comfyui_path not in sys.path:
            sys.path.insert(0, comfyui_path)
        
        # Set environment variables for ComfyUI
        os.environ['COMFY_BASE_DIR'] = comfyui_path
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('ComfyUIIntegrator')
        
        # Initialize API client
        self.api_client = ComfyUIAPI(server_url)
        
        # Store ComfyUI path for reference
        self.comfyui_path = comfyui_path
        
        # Add max prompt length
        self.MAX_PROMPT_LENGTH = 1024  # characters
        self.MAX_WORDS = 50  # maximum number of words
    
    def compress_prompt(self, prompt):
        """
        Compress a long prompt to improve generation speed
        
        Strategies:
        1. Remove extra whitespace
        2. Truncate to max length
        3. Reduce to most important words
        """
        # Remove extra whitespace
        prompt = re.sub(r'\s+', ' ', prompt).strip()
        
        # If prompt is short enough, return as-is
        if len(prompt) <= self.MAX_PROMPT_LENGTH:
            return prompt
        
        # Log original prompt length
        self.logger.info(f"Original prompt length: {len(prompt)} characters")
        
        # Truncate to max length
        prompt = prompt[:self.MAX_PROMPT_LENGTH]
        
        # Split into words and keep most important ones
        words = prompt.split()
        if len(words) > self.MAX_WORDS:
            # Keep first few words and last few words
            compressed_words = words[:10] + words[-10:]
            prompt = ' '.join(compressed_words)
        
        # Log compressed prompt
        self.logger.info(f"Compressed prompt length: {len(prompt)} characters")
        
        return prompt

    def generate_image(self, prompt: str, width: int = 1024, height: int = 1024, negative_prompt: str = "", steps: int = 30, cfg: float = 7.0, quality: float = 1.0, seed=None) -> Image.Image:
        """
        Generate an image using ComfyUI's SDXL workflow
        
        Args:
            prompt (str): Text prompt for image generation
            width (int): Width of the generated image
            height (int): Height of the generated image
            negative_prompt (str): Negative prompt to guide what not to generate
            steps (int): Number of sampling steps
            cfg (float): Classifier free guidance scale
            quality (float): Quality/denoise strength
            seed (int, optional): Seed for generation. If None, a random seed will be used.
            
        Returns:
            PIL.Image: Generated image
        """
        try:
            # Compress the prompt
            compressed_prompt = self.compress_prompt(prompt)
            compressed_negative_prompt = self.compress_prompt(negative_prompt) if negative_prompt else ""
            
            self.logger.info(f"Generating image with prompt: {prompt}")
            return self.api_client.generate_image(
                compressed_prompt, 
                negative_prompt=compressed_negative_prompt, 
                width=width, 
                height=height, 
                steps=steps, 
                cfg=cfg, 
                quality=quality,
                seed=seed
            )
        except Exception as e:
            self.logger.error(f"Image generation failed: {str(e)}")
            return None

class ComfyUIAPI:
    def __init__(self, server_url="http://127.0.0.1:8188"):
        self.server_url = server_url
        self.session = requests.Session()
        self.output_dir = os.path.join(os.path.dirname(__file__), "ComfyUI", "output")
        
        # Configure logging with DEBUG level
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("ComfyUIAPI")
        
        # Add a file handler to capture detailed logs
        log_file = os.path.join(os.path.dirname(__file__), "comfyui_api.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Optionally, reduce logging for requests library to avoid too much noise
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)

        # Add max prompt length
        self.MAX_PROMPT_LENGTH = 1024  # characters
        self.MAX_WORDS = 50  # maximum number of words

    def compress_prompt(self, prompt):
        """
        Compress a long prompt to improve generation speed
        
        Strategies:
        1. Remove extra whitespace
        2. Truncate to max length
        3. Reduce to most important words
        """
        # Remove extra whitespace
        prompt = re.sub(r'\s+', ' ', prompt).strip()
        
        # If prompt is short enough, return as-is
        if len(prompt) <= self.MAX_PROMPT_LENGTH:
            return prompt
        
        # Log original prompt length
        self.logger.info(f"Original prompt length: {len(prompt)} characters")
        
        # Truncate to max length
        prompt = prompt[:self.MAX_PROMPT_LENGTH]
        
        # Split into words and keep most important ones
        words = prompt.split()
        if len(words) > self.MAX_WORDS:
            # Keep first few words and last few words
            compressed_words = words[:10] + words[-10:]
            prompt = ' '.join(compressed_words)
        
        # Log compressed prompt
        self.logger.info(f"Compressed prompt length: {len(prompt)} characters")
        
        return prompt

    def setup_logging(self):
        self.logger = logging.getLogger("ComfyUIAPI")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def get_sdxl_workflow(self, prompt, negative_prompt="", width=1024, height=1024, steps=30, cfg=7.0, quality=1.0, seed=None):
        """Get the SDXL workflow with the specified parameters."""
        # Generate random seed if none specified
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
            
        self.logger.info(f"Using seed: {seed}")
        
        return {
            "3": {
                "inputs": {
                    "ckpt_name": "sdXL_v10.safetensors"
                },
                "class_type": "CheckpointLoaderSimple"
            },
            "4": {
                "inputs": {
                    "text": prompt,
                    "clip": ["3", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "5": {
                "inputs": {
                    "text": negative_prompt,
                    "clip": ["3", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "8": {
                "inputs": {
                    "samples": ["15", 0],
                    "vae": ["3", 2]
                },
                "class_type": "VAEDecode"
            },
            "9": {
                "inputs": {
                    "filename_prefix": "ComfyUI",
                    "images": ["8", 0]
                },
                "class_type": "SaveImage"
            },
            "15": {
                "inputs": {
                    "model": ["3", 0],
                    "positive": ["4", 0],
                    "negative": ["5", 0],
                    "latent_image": ["20", 0],
                    "seed": seed,
                    "steps": steps,
                    "cfg": cfg,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": quality,
                    "preview_method": "auto"
                },
                "class_type": "KSampler"
            },
            "20": {
                "inputs": {
                    "width": width,
                    "height": height,
                    "batch_size": 1
                },
                "class_type": "EmptyLatentImage"
            }
        }

    def queue_prompt(self, workflow):
        """
        Queue a workflow for image generation with comprehensive error handling
        
        Args:
            workflow (dict): ComfyUI workflow dictionary
        
        Returns:
            str or None: Prompt ID if successful, None otherwise
        """
        try:
            # Detailed logging of workflow
            self.logger.debug(f"Workflow to be queued: {json.dumps(workflow, indent=2)}")
            
            # Ensure the workflow is in the correct format
            payload = {"prompt": workflow}
            
            # Make the API request with timeout and error handling
            try:
                response = self.session.post(
                    f"{self.server_url}/prompt", 
                    json=payload,
                    headers={'Content-Type': 'application/json'},
                    timeout=10  # 10-second timeout
                )
            except requests.exceptions.RequestException as req_err:
                self.logger.error(f"Network error during prompt queueing: {req_err}")
                print(f"Network error: Unable to connect to ComfyUI server at {self.server_url}")
                return None
            
            # Check response status
            if response.status_code != 200:
                self.logger.error(f"HTTP Error: {response.status_code}")
                self.logger.error(f"Response content: {response.text}")
                print(f"Server returned error: {response.status_code}")
                return None
            
            # Try to parse JSON response
            try:
                response_data = response.json()
            except ValueError as json_err:
                self.logger.error(f"Failed to parse JSON response: {json_err}")
                self.logger.error(f"Raw response: {response.text}")
                print("Error: Received invalid JSON from server")
                return None
            
            # Validate prompt ID
            prompt_id = response_data.get("prompt_id")
            if not prompt_id:
                self.logger.error("No prompt_id found in server response")
                self.logger.error(f"Full response: {response_data}")
                print("Error: Server did not return a valid prompt ID")
                return None
            
            self.logger.info(f"Workflow queued successfully. Prompt ID: {prompt_id}")
            return prompt_id
        
        except Exception as e:
            self.logger.error(f"Unexpected error in queue_prompt: {e}", exc_info=True)
            print(f"Unexpected error: {e}")
            return None

    def get_image(self, prompt_id, timeout=300):
        """
        Retrieve generated image and track total generation time
        
        Args:
            prompt_id (str): ID of the queued workflow
            timeout (int): Maximum time to wait for image generation
        
        Returns:
            PIL.Image or None: Generated image or None if generation fails
        """
        start_time = time.time()
        
        self.logger.info(f"Starting image generation for prompt ID: {prompt_id}")
        print("\nGenerating image...")
        
        while time.time() - start_time < timeout:
            try:
                # Fetch history
                try:
                    history_response = self.session.get(
                        f"{self.server_url}/history/{prompt_id}", 
                        timeout=5
                    )
                    history_response.raise_for_status()
                    history = history_response.json()
                except (requests.RequestException, ValueError) as hist_err:
                    self.logger.warning(f"Failed to fetch history: {hist_err}")
                    time.sleep(1)
                    continue
                
                # Check if image is ready
                if prompt_id in history:
                    outputs = history[prompt_id].get("outputs", {})
                    for node_id, node_output in outputs.items():
                        if "images" in node_output:
                            image_data = node_output["images"][0]
                            
                            generation_time = time.time() - start_time
                            print(f"Image generated in {generation_time:.1f} seconds")
                            return self.download_image(image_data)
                
                time.sleep(0.5)
                
            except Exception as e:
                self.logger.error(f"Unexpected error during image generation: {e}", exc_info=True)
                print(f"\nError during generation: {e}")
                return None
        
        print("\nImage generation timed out")
        self.logger.error("Timeout waiting for image generation")
        return self.get_latest_image()

    def download_image(self, image_data):
        try:
            filename = image_data["filename"]
            subfolder = image_data.get("subfolder", "")
            image_url = f"{self.server_url}/view?filename={filename}&subfolder={subfolder}&type=output"
            response = self.session.get(image_url)
            response.raise_for_status()
            return Image.open(BytesIO(response.content))
        except Exception as e:
            self.logger.warning(f"API download failed: {str(e)}, trying filesystem")
            return self.get_latest_image()

    def get_latest_image(self):
        try:
            output_files = [
                os.path.join(self.output_dir, f)
                for f in os.listdir(self.output_dir)
                if f.startswith("MIDAS_Output") and f.endswith(".png")
            ]
            if not output_files:
                return None
            latest_file = max(output_files, key=os.path.getctime)
            return Image.open(latest_file)
        except Exception as e:
            self.logger.error(f"Failed to retrieve image: {str(e)}")
            return None

    def generate_image(self, prompt, negative_prompt="", width=1024, height=1024, steps=30, cfg=7.0, quality=1.0, seed=None):
        """
        Generate an image using the SDXL workflow
        
        Args:
            prompt (str): The prompt for image generation
            negative_prompt (str): What not to include in the image
            width (int): Image width
            height (int): Image height
            steps (int): Number of sampling steps
            cfg (float): Classifier free guidance scale
            quality (float): Quality/denoise strength
            seed (int, optional): Seed for generation. If None, a random seed will be used.
            
        Returns:
            PIL.Image: Generated image
        """
        try:
            # Compress the prompt
            compressed_prompt = self.compress_prompt(prompt)
            compressed_negative_prompt = self.compress_prompt(negative_prompt) if negative_prompt else ""
            
            # Get the workflow with parameters
            workflow = self.get_sdxl_workflow(
                prompt=compressed_prompt,
                negative_prompt=compressed_negative_prompt,
                width=width,
                height=height,
                steps=steps,
                cfg=cfg,
                quality=quality,
                seed=seed
            )
            
            # Queue the prompt
            p = self.queue_prompt(workflow)
            
            # Get the image output
            return self.get_image(p)
            
        except Exception as e:
            self.logger.error(f"Failed to generate image: {str(e)}")
            return None