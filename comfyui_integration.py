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
import glob

def setup_logging(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def compress_prompt(prompt):
    # Remove extra whitespace
    prompt = re.sub(r'\s+', ' ', prompt).strip()
    
    # If prompt is short enough, return as-is
    if len(prompt) <= 1024:
        return prompt
    
    # Log original prompt length
    logging.info(f"Original prompt length: {len(prompt)} characters")
    
    # Truncate to max length
    prompt = prompt[:1024]
    
    # Split into words and keep most important ones
    words = prompt.split()
    if len(words) > 50:
        # Keep first few words and last few words
        compressed_words = words[:10] + words[-10:]
        prompt = ' '.join(compressed_words)
    
    # Log compressed prompt
    logging.info(f"Compressed prompt length: {len(prompt)} characters")
    
    return prompt

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
        self.logger = setup_logging('ComfyUIIntegrator')
        
        # Initialize API client
        self.api_client = ComfyUIAPI(server_url)
        
        # Store ComfyUI path for reference
        self.comfyui_path = comfyui_path
        
        # Add max prompt length
        self.MAX_PROMPT_LENGTH = 1024  # characters
        self.MAX_WORDS = 50  # maximum number of words
    
    def generate_image(self, prompt: str, width: int = 1024, height: int = 1024, negative_prompt: str = "", steps: int = 30, cfg: float = 7.0, quality: float = 1.0, seed=None, timeout=600, workflow_name=None) -> Image.Image:
        """
        Generate an image using ComfyUI's workflow
        
        Args:
            prompt (str): Text prompt for image generation
            width (int): Width of the generated image
            height (int): Height of the generated image
            negative_prompt (str): Negative prompt to guide what not to generate
            steps (int): Number of sampling steps
            cfg (float): Classifier free guidance scale
            quality (float): Quality/denoise strength
            seed (int, optional): Seed for generation. If None, a random seed will be used.
            timeout (int, optional): Maximum time to wait for image generation in seconds. Default is 600 (10 minutes).
            workflow_name (str, optional): Name of the workflow to use. If None, the default SDXL workflow will be used.
            
        Returns:
            PIL.Image: Generated image
        """
        try:
            # Compress the prompt
            compressed_prompt = compress_prompt(prompt)
            compressed_negative_prompt = compress_prompt(negative_prompt) if negative_prompt else ""
            
            self.logger.info(f"Generating image with prompt: {prompt}")
            return self.api_client.generate_image(
                compressed_prompt, 
                negative_prompt=compressed_negative_prompt, 
                width=width, 
                height=height, 
                steps=steps, 
                cfg=cfg, 
                quality=quality,
                seed=seed,
                timeout=timeout,
                workflow_name=workflow_name
            )
        except Exception as e:
            self.logger.error(f"Image generation failed: {str(e)}")
            return None

class ComfyUIAPI:
    def __init__(self, server_url="http://127.0.0.1:8188"):
        self.server_url = server_url
        self.session = requests.Session()
        self.output_dir = os.path.join(os.path.dirname(__file__), "ComfyUI", "output")
        self.workflows_dir = os.path.join(os.path.dirname(__file__), "workflows")
        
        # Configure logging with DEBUG level
        self.logger = setup_logging('ComfyUIAPI')
        
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

    def get_available_workflows(self):
        """
        Get a list of available workflow names from the workflows directory
        
        Returns:
            list: List of workflow names without file extensions
        """
        try:
            # Ensure workflows directory exists
            if not os.path.exists(self.workflows_dir):
                os.makedirs(self.workflows_dir, exist_ok=True)
                self.logger.info(f"Created workflows directory at {self.workflows_dir}")
                return []
            
            # Get all JSON files in the workflows directory
            workflow_files = glob.glob(os.path.join(self.workflows_dir, "*.json"))
            
            # Extract workflow names without file extensions
            workflow_names = [os.path.splitext(os.path.basename(f))[0] for f in workflow_files]
            
            self.logger.info(f"Found {len(workflow_names)} workflows: {', '.join(workflow_names)}")
            return workflow_names
        
        except Exception as e:
            self.logger.error(f"Error getting available workflows: {str(e)}")
            return []

    def load_workflow(self, workflow_name):
        """
        Load a workflow from the workflows directory
        
        Args:
            workflow_name (str): Name of the workflow file without extension
            
        Returns:
            dict: Workflow JSON data or None if not found
        """
        try:
            # Ensure workflows directory exists
            if not os.path.exists(self.workflows_dir):
                self.logger.error(f"Workflows directory does not exist: {self.workflows_dir}")
                return None
            
            # Normalize workflow name
            if workflow_name.startswith("Workflow: "):
                workflow_name = workflow_name[len("Workflow: "):]
            
            # Detailed logging of workflow loading attempt
            self.logger.info("=" * 50)
            self.logger.info(f"ATTEMPTING TO LOAD WORKFLOW: {workflow_name}")
            self.logger.info("=" * 50)
            
            # List all files in the workflows directory
            try:
                dir_contents = os.listdir(self.workflows_dir)
                self.logger.info(f"Contents of workflows directory: {dir_contents}")
            except Exception as dir_err:
                self.logger.error(f"Error listing workflows directory: {dir_err}")
            
            # Find all JSON workflow files
            available_workflows = glob.glob(os.path.join(self.workflows_dir, "*.json"))
            self.logger.info(f"All available workflow files: {available_workflows}")
            
            # Try multiple variations of the workflow name
            workflow_variations = [
                workflow_name,  # Original name
                workflow_name.replace(" ", "_"),  # Replace spaces with underscores
                workflow_name.replace(" ", ""),  # Remove spaces
                workflow_name.lower(),  # Lowercase
                workflow_name.replace(" ", "_").lower(),  # Lowercase with underscores
                workflow_name.replace(" ", "").lower(),  # Lowercase without spaces
            ]
            
            # Find the first matching workflow file
            matching_workflows = []
            for variation in workflow_variations:
                matching_workflows = [
                    f for f in available_workflows 
                    if os.path.splitext(os.path.basename(f))[0].lower() == variation.lower()
                ]
                if matching_workflows:
                    break
            
            # If no match found, log error and return None
            if not matching_workflows:
                self.logger.error(f"No workflow found matching: {workflow_name}")
                
                # Print out potential close matches
                def find_close_matches(name, options):
                    import difflib
                    return difflib.get_close_matches(name, options, n=3, cutoff=0.6)
                
                close_matches = find_close_matches(workflow_name, 
                    [os.path.splitext(os.path.basename(f))[0] for f in available_workflows])
                
                if close_matches:
                    self.logger.info(f"Possible close matches: {close_matches}")
                
                return None
            
            # Use the first matching workflow
            workflow_path = matching_workflows[0]
            
            # Detailed logging of workflow path
            self.logger.info(f"Found matching workflow: {workflow_path}")
            
            # Read and parse the workflow file
            try:
                with open(workflow_path, 'r') as f:
                    workflow_data = json.load(f)
                
                # Detailed logging of workflow data
                self.logger.info(f"Successfully loaded workflow from: {workflow_path}")
                self.logger.debug(f"Workflow Structure: {json.dumps(workflow_data, indent=2)}")
                
                return workflow_data
            
            except json.JSONDecodeError as json_err:
                self.logger.error(f"JSON parsing error in workflow file {workflow_path}: {json_err}")
                return None
            except IOError as io_err:
                self.logger.error(f"IO error reading workflow file {workflow_path}: {io_err}")
                return None
        
        except Exception as e:
            self.logger.error(f"Unexpected error loading workflow {workflow_name}: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def update_workflow_with_parameters(self, workflow, prompt, negative_prompt="", width=1024, height=1024, steps=15, cfg=2.0, quality=1.0, seed=None):
        """
        Update a workflow with the specified parameters
        
        Args:
            workflow (dict): Workflow data
            prompt (str): Text prompt
            negative_prompt (str): Negative prompt
            width (int): Image width
            height (int): Image height
            steps (int): Number of sampling steps
            cfg (float): Classifier free guidance scale
            quality (float): Quality/denoise strength
            seed (int): Seed for generation
            
        Returns:
            dict: Updated workflow
        """
        try:
            # Generate random seed if none specified
            if seed is None:
                seed = random.randint(0, 2**32 - 1)
                
            self.logger.info(f"Using seed: {seed}")
            
            # Make a copy of the workflow to avoid modifying the original
            updated_workflow = workflow.copy()
            
            # Find nodes by class type
            for node_id, node in updated_workflow.items():
                # Update positive prompt in CLIPTextEncode nodes
                if node.get("class_type") == "CLIPTextEncode" and "_meta" in node and "title" in node["_meta"] and "Positive" in node["_meta"]["title"]:
                    node["inputs"]["text"] = prompt
                    self.logger.info(f"Updated positive prompt in node {node_id}")
                
                # Update negative prompt in CLIPTextEncode nodes
                elif node.get("class_type") == "CLIPTextEncode" and "_meta" in node and "title" in node["_meta"] and "Negative" in node["_meta"]["title"]:
                    node["inputs"]["text"] = negative_prompt
                    self.logger.info(f"Updated negative prompt in node {node_id}")
                
                # Update seed in RandomNoise or similar nodes
                elif node.get("class_type") in ["RandomNoise", "KSampler"] and "noise_seed" in node["inputs"]:
                    node["inputs"]["noise_seed"] = seed
                    self.logger.info(f"Updated seed in node {node_id}")
                elif node.get("class_type") in ["KSampler"] and "seed" in node["inputs"]:
                    node["inputs"]["seed"] = seed
                    self.logger.info(f"Updated seed in node {node_id}")
                
                # Update steps in KSampler or similar nodes
                elif "steps" in node.get("inputs", {}) and node.get("class_type") in ["KSampler", "BasicScheduler"]:
                    node["inputs"]["steps"] = steps
                    self.logger.info(f"Updated steps in node {node_id}")
                
                # Update cfg in KSampler nodes
                elif "cfg" in node.get("inputs", {}) and node.get("class_type") in ["KSampler"]:
                    node["inputs"]["cfg"] = cfg
                    self.logger.info(f"Updated cfg in node {node_id}")
                
                # Update denoise/quality in KSampler or BasicScheduler nodes
                elif "denoise" in node.get("inputs", {}) and node.get("class_type") in ["KSampler", "BasicScheduler"]:
                    node["inputs"]["denoise"] = quality
                    self.logger.info(f"Updated denoise in node {node_id}")
                
                # Update width and height in EmptyLatentImage or similar nodes
                elif node.get("class_type") in ["EmptyLatentImage", "EmptySD3LatentImage"] and "width" in node["inputs"] and "height" in node["inputs"]:
                    node["inputs"]["width"] = width
                    node["inputs"]["height"] = height
                    self.logger.info(f"Updated dimensions in node {node_id}")
            
            return updated_workflow
        
        except Exception as e:
            self.logger.error(f"Error updating workflow parameters: {str(e)}")
            return workflow  # Return original workflow if update fails

    def get_sdxl_workflow(self, prompt, negative_prompt="", width=1024, height=1024, steps=15, cfg=7.0, quality=1.0, seed=None):
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
        poll_interval = 1.0  # Start with 1 second polling
        max_poll_interval = 2.0  # Maximum polling interval
        
        self.logger.info(f"Starting image generation for prompt ID: {prompt_id}")
        print("\nGenerating image...")
        
        while time.time() - start_time < timeout:
            try:
                # Fetch history
                history_response = self.session.get(
                    f"{self.server_url}/history/{prompt_id}", 
                    timeout=5
                )
                history_response.raise_for_status()
                history = history_response.json()
                
                # Check if image is ready
                if prompt_id in history:
                    outputs = history[prompt_id].get("outputs", {})
                    for node_id, node_output in outputs.items():
                        if "images" in node_output:
                            image_data = node_output["images"][0]
                            
                            generation_time = time.time() - start_time
                            print(f"Image generated in {generation_time:.1f} seconds")
                            
                            # Try API download first
                            try:
                                filename = image_data["filename"]
                                subfolder = image_data.get("subfolder", "")
                                image_url = f"{self.server_url}/view?filename={filename}&subfolder={subfolder}&type=output"
                                response = self.session.get(image_url, timeout=5)
                                response.raise_for_status()
                                return Image.open(BytesIO(response.content))
                            except Exception as download_err:
                                self.logger.warning(f"API download failed: {str(download_err)}, using filesystem")
                                # Small delay before filesystem check
                                time.sleep(0.5)
                                return self.get_latest_image()
                
                # Adaptive polling interval
                time.sleep(min(poll_interval, max_poll_interval))
                poll_interval *= 1.5  # Increase polling interval
                
            except requests.RequestException as e:
                self.logger.warning(f"Request failed: {e}")
                time.sleep(1)
                continue
            except Exception as e:
                self.logger.error(f"Unexpected error during image generation: {e}", exc_info=True)
                print(f"\nError during generation: {e}")
                return None
        
        print("\nImage generation timed out")
        self.logger.error("Timeout waiting for image generation")
        return self.get_latest_image()

    def generate_image(self, prompt: str, width: int = 1024, height: int = 1024, negative_prompt: str = "", steps: int = 30, cfg: float = 7.0, quality: float = 1.0, seed=None, timeout=600, workflow_name=None) -> Image.Image:
        """
        Generate an image using the specified workflow or default SDXL workflow
        
        Args:
            prompt (str): The prompt for image generation
            negative_prompt (str): What not to include in the image
            width (int): Image width
            height (int): Image height
            steps (int): Number of sampling steps
            cfg (float): Classifier free guidance scale
            quality (float): Quality/denoise strength
            seed (int, optional): Seed for generation. If None, a random seed will be used.
            timeout (int, optional): Maximum time to wait for image generation in seconds. Default is 600 (10 minutes).
            workflow_name (str, optional): Name of the workflow to use. If None, the default SDXL workflow will be used.
            
        Returns:
            PIL.Image: Generated image
        """
        try:
            # Detailed logging of ALL input parameters
            self.logger.info("=" * 50)
            self.logger.info("GENERATE IMAGE CALLED")
            self.logger.info("=" * 50)
            self.logger.info(f"Full Input Parameters:")
            self.logger.info(f"Prompt: {prompt}")
            self.logger.info(f"Workflow Name: {workflow_name}")
            self.logger.info(f"Width: {width}, Height: {height}")
            self.logger.info(f"Negative Prompt: {negative_prompt}")
            self.logger.info(f"Steps: {steps}")
            self.logger.info(f"CFG: {cfg}")
            self.logger.info(f"Quality: {quality}")
            self.logger.info(f"Seed: {seed}")
            
            # Normalize workflow name
            if workflow_name and workflow_name.startswith("Workflow: "):
                workflow_name = workflow_name[len("Workflow: "):]
            
            # Detailed logging of workflow name normalization
            self.logger.info(f"Normalized Workflow Name: {workflow_name}")
            
            # Compress the prompt
            compressed_prompt = compress_prompt(prompt)
            compressed_negative_prompt = compress_prompt(negative_prompt) if negative_prompt else ""
            
            # Get the workflow with parameters
            if workflow_name:
                # Log all available workflows
                available_workflows = glob.glob(os.path.join(self.workflows_dir, "*.json"))
                self.logger.info(f"All Available Workflows: {available_workflows}")
                
                # Log the full path of the workflow
                workflow_path = os.path.join(self.workflows_dir, f"{workflow_name}.json")
                self.logger.info(f"Attempting to load workflow from: {workflow_path}")
                
                # Detailed workflow loading
                workflow_data = self.load_workflow(workflow_name)
                if workflow_data:
                    workflow = self.update_workflow_with_parameters(
                        workflow_data,
                        prompt=compressed_prompt,
                        negative_prompt=compressed_negative_prompt,
                        width=width,
                        height=height,
                        steps=steps,
                        cfg=cfg,
                        quality=quality,
                        seed=seed
                    )
                    self.logger.info(f"Successfully loaded and updated custom workflow: {workflow_name}")
                    
                    # Log the structure of the updated workflow
                    self.logger.debug(f"Updated Workflow Structure: {json.dumps(workflow, indent=2)}")
                else:
                    # Fallback to default SDXL workflow if custom workflow not found
                    self.logger.warning(f"Custom workflow {workflow_name} not found, falling back to SDXL")
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
            else:
                # Use default SDXL workflow
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
            
            # Log the final workflow type
            self.logger.info(f"Final Workflow Type: {'Custom' if workflow_name else 'SDXL'}")
            
            # Queue the prompt
            prompt_id = self.queue_prompt(workflow)
            if not prompt_id:
                self.logger.error("Failed to queue workflow")
                return None
            
            # Get the image output
            return self.get_image(prompt_id, timeout=timeout)
            
        except Exception as e:
            self.logger.error(f"Failed to generate image: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

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