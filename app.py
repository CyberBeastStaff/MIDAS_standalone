import os
import json
import uuid
import shutil
import sqlite3
import random
import pandas as pd
import traceback
from datetime import datetime
import tempfile
import requests
import threading
import atexit
import base64
import io
import configparser
import time
import glob
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

import gradio as gr
import ollama
from flask import Flask
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain.embeddings.base import Embeddings

class TfidfEmbeddings(Embeddings):
    """
    A simple embedding class that uses TF-IDF vectorization from scikit-learn.
    Implements the Embeddings interface for compatibility with LangChain.
    """
    
    def __init__(self, **kwargs):
        """Initialize the TF-IDF vectorizer."""
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            max_features=354,  # Set to match the dimensionality of HuggingFaceEmbeddings
            stop_words='english'
        )
        self.fitted = False
        self.documents = []
        self.output_dim = 354  # Fixed dimension to match existing embeddings
        
    def _fit_vectorizer(self, texts):
        """Fit the vectorizer on the provided texts."""
        if not self.fitted:
            self.vectorizer.fit(texts)
            self.fitted = True
            self.documents = texts
    
    def embed_documents(self, texts):
        """
        Embed a list of documents using TF-IDF.
        
        Args:
            texts: List of strings to embed
            
        Returns:
            List of embeddings, one for each text
        """
        # Ensure vectorizer is fitted
        all_texts = self.documents + texts if self.fitted else texts
        self._fit_vectorizer(all_texts)
        
        # Transform texts to TF-IDF vectors
        tfidf_matrix = self.vectorizer.transform(texts)
        
        # Convert sparse matrix to dense and normalize
        embeddings = []
        for i in range(tfidf_matrix.shape[0]):
            vec = tfidf_matrix[i].toarray()[0]
            
            # Ensure the vector has exactly the required dimensions
            if len(vec) < self.output_dim:
                # Pad with zeros if needed
                vec = np.pad(vec, (0, self.output_dim - len(vec)))
            elif len(vec) > self.output_dim:
                # Truncate if needed
                vec = vec[:self.output_dim]
                
            # Normalize to unit length
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            embeddings.append(vec.tolist())
        
        return embeddings
    
    def embed_query(self, text):
        """
        Embed a query using TF-IDF.
        
        Args:
            text: String to embed
            
        Returns:
            Embedding for the text
        """
        return self.embed_documents([text])[0]

# Load configuration
config = configparser.ConfigParser()
config.read('config.ini')

# Initialize Flask app
app = Flask(__name__)

# Constants
DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant. Answer questions accurately and concisely."""

# Configure Ollama client
ollama_client = ollama.Client(host='http://localhost:11434')

# Database setup
def migrate_database():
    """Perform database migrations"""
    conn = sqlite3.connect('conversations.db')
    c = conn.cursor()
    
    try:
        # Check if deleted column exists
        c.execute("PRAGMA table_info(documents)")
        columns = [col[1] for col in c.fetchall()]
        
        if 'deleted' not in columns:
            print("Adding 'deleted' column to documents table")
            c.execute('ALTER TABLE documents ADD COLUMN deleted INTEGER DEFAULT 0')
            conn.commit()
            
    except Exception as e:
        print(f"Error during migration: {e}")
        traceback.print_exc()
    finally:
        conn.close()

def init_db():
    """Initialize the database with required tables"""
    conn = sqlite3.connect('conversations.db')
    c = conn.cursor()
    
    # Create conversations table
    c.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_used_model TEXT
        )
    ''')
    
    # Create messages table
    c.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER,
            model TEXT,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        )
    ''')
    
    # Create documents table with deleted flag
    c.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            original_name TEXT NOT NULL,
            file_type TEXT NOT NULL,
            embedding_path TEXT,
            conversation_id INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            deleted INTEGER DEFAULT 0,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        )
    ''')
    
    # Create settings table
    c.execute('''
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    ''')
    
    # Check if default model exists
    c.execute('SELECT value FROM settings WHERE key = "default_model"')
    result = c.fetchone()
    
    if not result:
        # Set default model if not exists
        c.execute('''
            INSERT INTO settings (key, value) 
            VALUES ('default_model', 'llama3.1:8b')
        ''')
        print("Initialized default model setting to llama3.1:8b")
    else:
        print(f"Using existing default model: {result[0]}")
    
    # Commit changes and close connection
    conn.commit()
    conn.close()
    
    # Run migrations after creating tables
    migrate_database()

def init_bots_db():
    """Initialize the database with required tables"""
    conn = sqlite3.connect('bots.db')
    c = conn.cursor()
    
    # Create bots table
    c.execute('''CREATE TABLE IF NOT EXISTS bots
                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE,
                base_model TEXT,
                system_prompt TEXT,
                temperature REAL,
                max_tokens INTEGER,
                top_p REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    
    # Create bot_documents table
    c.execute('''CREATE TABLE IF NOT EXISTS bot_documents
                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                bot_id INTEGER,
                filename TEXT,
                original_name TEXT,
                file_type TEXT,
                embedding_path TEXT,
                chunk_size INTEGER DEFAULT 1000,
                chunk_overlap INTEGER DEFAULT 200,
                uploaded_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(bot_id) REFERENCES bots(id))''')
    
    conn.commit()
    conn.close()

def save_bot(bot_name, base_model, system_prompt, temperature, max_tokens, top_p):
    """
    Save or update a bot configuration
    
    Args:
        bot_name (str): Name of the bot
        base_model (str): Base model for the bot
        system_prompt (str): System prompt for the bot
        temperature (float): Temperature parameter
        max_tokens (int): Maximum tokens parameter
        top_p (float): Top P parameter
    
    Returns:
        tuple: (success_status, message)
    """
    try:
        conn = sqlite3.connect('bots.db')
        c = conn.cursor()
        
        # Validate input
        if not bot_name:
            return False, "Bot name cannot be empty"
            
        # Convert parameters to appropriate types
        try:
            temperature = float(temperature)
            max_tokens = int(max_tokens)
            top_p = float(top_p)
        except (ValueError, TypeError) as e:
            return False, f"Invalid parameter values: {str(e)}"
            
        # Validate parameter ranges
        if not (0.0 <= temperature <= 2.0):
            return False, "Temperature must be between 0.0 and 2.0"
        if not (0.0 <= top_p <= 1.0):
            return False, "Top P must be between 0.0 and 1.0"
        if max_tokens <= 0:
            return False, "Max tokens must be greater than 0"
        
        # Check if bot already exists
        c.execute('SELECT id FROM bots WHERE name = ?', (bot_name,))
        existing_bot = c.fetchone()
        
        if existing_bot:
            # Update existing bot
            c.execute('''UPDATE bots 
                         SET base_model = ?, 
                             system_prompt = ?, 
                             temperature = ?, 
                             max_tokens = ?, 
                             top_p = ?,
                             updated_at = CURRENT_TIMESTAMP
                         WHERE name = ?''', 
                      (base_model, system_prompt, temperature, max_tokens, top_p, bot_name))
            message = f"Bot '{bot_name}' updated successfully"
        else:
            # Insert new bot
            c.execute('''INSERT INTO bots 
                         (name, base_model, system_prompt, temperature, max_tokens, top_p)
                         VALUES (?, ?, ?, ?, ?, ?)''', 
                      (bot_name, base_model, system_prompt, temperature, max_tokens, top_p))
            message = f"Bot '{bot_name}' created successfully"
        
        conn.commit()
        conn.close()
        return True, message
    
    except sqlite3.IntegrityError:
        return False, f"A bot with the name '{bot_name}' already exists"
    except Exception as e:
        print(f"Error saving bot: {e}")
        return False, f"Error saving bot: {str(e)}"

def get_bot_configurations():
    """
    Retrieve all bot configurations
    
    Args:
        None
    
    Returns:
        list: List of dictionaries containing bot configurations
    """
    try:
        conn = sqlite3.connect('bots.db')
        c = conn.cursor()
        
        c.execute('''SELECT name, base_model, system_prompt, 
                            temperature, max_tokens, top_p 
                     FROM bots 
                     ORDER BY created_at DESC''')
        
        columns = ['name', 'base_model', 'system_prompt', 
                   'temperature', 'max_tokens', 'top_p']
        
        bots = [dict(zip(columns, row)) for row in c.fetchall()]
        
        conn.close()
        return bots
    
    except Exception as e:
        print(f"Error retrieving bot configurations: {e}")
        return []

def get_available_models():
    """
    Get list of available models from Ollama or configuration file
    
    Args:
        None
    
    Returns:
        list: List of model names without version suffixes
    """
    try:
        # Get models using ollama list command
        models = []
        response = requests.get('http://localhost:11434/api/tags')
        if response.status_code == 200:
            data = response.json()
            if 'models' in data:
                # Extract base model names without version
                models = [model['name'].split(':')[0] for model in data['models']]
        
        # If no models found, try reading from configuration
        if not models:
            # Read default models from configuration or environment
            default_models_str = os.environ.get('MIDAS_DEFAULT_MODELS', 
                                                config.get('models', 'default_models', 
                                                          fallback='["phi3.5", "mistral"]'))
            try:
                models = json.loads(default_models_str)
            except (json.JSONDecodeError, TypeError):
                # Fallback to a minimal set if parsing fails
                models = ["phi3.5", "mistral"]
        
        # Remove duplicates and sort
        models = sorted(set(models))
        
        return models
        
    except Exception as e:
        print(f"Error getting models: {e}")
        # Return minimal default models if everything fails
        return ["phi3.5", "mistral"]

def get_model_display_name(model_name):
    """
    Get display name for a model (without version)
    
    Args:
        model_name (str): Full model name with version
    
    Returns:
        str: Model name without version
    """
    return model_name.split(':')[0]

def get_model_version(model_name):
    """
    Get the correct version suffix for a model
    
    Args:
        model_name (str): Model to use for the conversation
    
    Returns:
        str: Full model name with correct version
    """
    # Model-specific version mapping
    version_map = {
        "llama3.1": "8b",
        "phi3.5": "latest",
        "mistral": "latest",
        "codellama": "latest",
        "deepseek-r1": "7b"
    }
    
    # If model already has a version, return as is
    if ':' in model_name:
        return model_name
    
    # Get base model name
    base_model = model_name.split(':')[0]
    
    # Return with correct version
    version = version_map.get(base_model, "latest")
    return f"{base_model}:{version}"

def get_available_ollama_models():
    """
    Retrieve detailed list of available Ollama models
    
    Args:
        None
    
    Returns:
        list: List of dictionaries containing model details
    """
    try:
        response = requests.get('http://localhost:11434/api/tags')
        if response.status_code == 200:
            models = response.json().get('models', [])
            
            # Transform model data for display
            formatted_models = []
            for model in models:
                formatted_models.append({
                    'Model Name': model.get('name', 'Unknown'),
                    'Size': f"{model.get('size', 0) / (1024 * 1024 * 1024):.2f} GB",
                    'Actions': 'Delete'  # Placeholder for future actions
                })
            
            return formatted_models
        else:
            print(f"Error fetching Ollama models: {response.text}")
            return []
    except Exception as e:
        print(f"Error retrieving Ollama models: {e}")
        return []

def get_bot_by_name(bot_name):
    """
    Retrieve a specific bot configuration by name
    
    Args:
        bot_name (str): Name of the bot to retrieve
    
    Returns:
        dict: Bot configuration or None if not found
    """
    try:
        conn = sqlite3.connect('bots.db')
        c = conn.cursor()
        
        c.execute('''SELECT name, base_model, system_prompt, 
                            temperature, max_tokens, top_p 
                     FROM bots 
                     WHERE name = ?''', (bot_name,))
        
        columns = ['name', 'base_model', 'system_prompt', 
                   'temperature', 'max_tokens', 'top_p']
        
        result = c.fetchone()
        conn.close()
        
        return dict(zip(columns, result)) if result else None
    
    except Exception as e:
        print(f"Error retrieving bot configuration: {e}")
        return None

def get_bot_names():
    """
    Retrieve names of all saved bots
    
    Args:
        None
    
    Returns:
        list: List of bot names
    """
    try:
        conn = sqlite3.connect('bots.db')
        c = conn.cursor()
        
        c.execute('SELECT name FROM bots ORDER BY created_at DESC')
        
        bot_names = [row[0] for row in c.fetchall()]
        
        conn.close()
        return bot_names
    
    except Exception as e:
        print(f"Error retrieving bot names: {e}")
        return []

def generate_response(message, model, chat_history, conv_id, context=None):
    """
    Generate a response using the specified model or bot configuration
    
    Args:
        message (str): User's input message
        model (str): Model to use for the conversation
        chat_history (list): Conversation history
        conv_id (str): Conversation ID
        context (str, optional): RAG context
    
    Returns:
        tuple: Empty string, updated chat history, conversation ID
    """
    global generation_active
    generation_active = True
    try:
        # Check if this is a bot or a base model
        bot_config = get_bot_by_name(model)
        if bot_config:
            # Get context from bot's knowledge base
            bot_context = get_bot_knowledge_context(model, message)
            
            # Combine RAG context with bot's knowledge base context
            combined_context = ""
            if context:
                combined_context += f"Context from uploaded documents:\n{context}\n\n"
            if bot_context:
                combined_context += f"Context from bot's knowledge base:\n{bot_context}\n\n"
            
            # Use bot configuration
            base_model = bot_config['base_model']
            system_prompt = bot_config['system_prompt']
            temperature = bot_config['temperature']
            max_tokens = bot_config['max_tokens']
            top_p = bot_config['top_p']
        else:
            # Use default parameters for base model
            base_model = get_model_version(model)
            system_prompt = DEFAULT_SYSTEM_PROMPT
            temperature = 0.7
            max_tokens = 8196
            top_p = 0.9
            combined_context = context if context else ""
        
        # Ensure model name has correct version
        base_model = get_model_version(base_model)
        
        # Create messages array with system prompt and context
        messages = []
        
        # Add system prompt
        messages.append({"role": "system", "content": system_prompt})
        
        # Add context if available
        if combined_context:
            messages.append({
                "role": "system",
                "content": f"""Use the following context to help answer the user's question. The context contains relevant information from documents:

{combined_context}

Remember to:
1. Use the context to provide accurate and relevant information
2. If the context doesn't fully answer the question, combine it with your general knowledge
3. Always cite specific information from the context when using it
4. If the context is not relevant to the question, rely on your general knowledge instead

Please answer the user's question:"""
            })
        
        # Add chat history
        for msg in chat_history:
            messages.append({"role": "user", "content": msg[0]})
            if msg[1]:  # Only add assistant message if it exists
                messages.append({"role": "assistant", "content": msg[1]})
        
        # Add current message
        messages.append({"role": "user", "content": message})
        
        # Create new conversation if needed
        is_first_message = len(chat_history) == 0
        if not conv_id:
            conv_id = create_conversation(message, model)
        
        # Generate response
        response = ""
        response_lines = []
        thinking_mode = False
        final_response = ""
        
        try:
            print(f"Generating response with model: {base_model}")
            print(f"Using parameters: temperature={temperature}, max_tokens={max_tokens}, top_p={top_p}")
            
            # Convert parameters to appropriate types
            temperature = float(temperature)
            max_tokens = int(max_tokens)
            top_p = float(top_p)
            
            for chunk in ollama_client.chat(
                model=base_model,
                messages=messages,
                stream=True,
                options={
                    "temperature": temperature,
                    "top_p": top_p,
                    "num_predict": max_tokens,
                    "stop": ["</s>", "user:", "User:", "assistant:", "Assistant:"]
                }
            ):
                if not generation_active:
                    break
                if chunk['message']['content']:
                    response += chunk['message']['content']
                    response_lines.append(chunk['message']['content'])
                    
                    # Check if we're in a thinking model response
                    if "<think>" in response and not thinking_mode:
                        thinking_mode = True
                        # Extract thinking content
                        thinking_content = response.split("<think>", 1)[1].strip()
                        current_response = "> **Thinking Process:**\n> \n> " + thinking_content.replace('\n', '\n> ')
                    elif thinking_mode and "</think>" in response:
                        # Extract thinking content
                        thinking_part = response.split("<think>", 1)[1].split("</think>", 1)[0].strip()
                        # Extract final response after </think> tag
                        final_part = response.split("</think>", 1)[1].strip()
                        # Display thinking and final response using markdown quotes for thinking
                        current_response = "> **Thinking Process:**\n> \n> " + thinking_part.replace('\n', '\n> ') + "\n\n" + final_part
                        final_response = final_part  # Save the final response
                    elif thinking_mode:
                        # Still in thinking mode, update thinking content
                        thinking_content = response.split("<think>", 1)[1].strip()
                        current_response = "> **Thinking Process:**\n> \n> " + thinking_content.replace('\n', '\n> ')
                    else:
                        # Regular response
                        current_response = response
                    
                    # Update chat history with current response
                    updated_history = chat_history + [(message, current_response)]
                    
                    yield current_response, updated_history, conv_id
            
            # Save the final message to the database once streaming is complete
            save_message(conv_id, model, "user", message)
            
            # For thinking models, save both thinking process and final response in one message
            if thinking_mode and final_response:
                # Extract the thinking process
                thinking_content = response.split("<think>", 1)[1].split("</think>", 1)[0].strip()
                # Format with markdown quotes for thinking and regular text for response
                thinking_formatted = thinking_content.replace('\n', '\n> ')
                combined_response = f"> **Thinking Process:**\n> \n> {thinking_formatted}\n\n{final_response}"
                # Save as a single assistant message
                save_message(conv_id, model, "assistant", combined_response)
            else:
                # Save regular response
                save_message(conv_id, model, "assistant", response)
        except Exception as e:
            print(f"Error during response generation: {e}")
            error_msg = "I apologize, but I encountered an error while generating the response. Please try again."
            updated_history = chat_history + [(message, error_msg)]
            yield error_msg, updated_history, conv_id
            return
        
        # Save final message pair
        # For thinking models, preserve the thinking process in the final response
        if thinking_mode and final_response:
            thinking_formatted = thinking_content.replace('\n', '\n> ')
            combined_response = f"> **Thinking Process:**\n> \n> {thinking_formatted}\n\n{final_response}"
            response = combined_response

        # Generate title for the conversation if it's the first message
        conn = sqlite3.connect('conversations.db')
        c = conn.cursor()
        c.execute('SELECT title FROM conversations WHERE id = ?', (conv_id,))
        current_title = c.fetchone()
        conn.close()
        
        # Only generate a title if this is the first message and the current title is empty or "Untitled Conversation"
        if is_first_message and current_title and (not current_title[0] or current_title[0].strip() == "Untitled Conversation"):
            title = generate_conversation_title(message, response, model)
            update_conversation_title(conv_id, title)
            update_conversation_model(model, conv_id)
        updated_history = chat_history + [(message, response)]
        yield response, updated_history, conv_id
    except Exception as e:
        print(f"Error generating response: {e}")
        traceback.print_exc()
        error_msg = "I apologize, but I encountered an error while processing your request."
        updated_history = chat_history + [(message, error_msg)]
        yield error_msg, updated_history, conv_id

def save_message(conv_id, model, role, content, full_history=None):
    """
    Save a message to the conversation database
    
    Args:
        conv_id (int): Conversation ID
        model (str): Model used
        role (str): Role of the message (user/assistant)
        content (str): Message content
        full_history (list): Full conversation history
    """
    try:
        conn = sqlite3.connect('conversations.db')
        cursor = conn.cursor()
        
        # Ensure messages table exists
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER,
                model TEXT,
                role TEXT,
                content TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        if full_history:
            # Remove only the last two messages instead of clearing all
            cursor.execute('''
                DELETE FROM messages
                WHERE id IN (
                    SELECT id
                    FROM messages
                    WHERE conversation_id = ?
                    ORDER BY id DESC
                    LIMIT 2
                )
            ''', (conv_id,))
        else:
            # Insert the message
            cursor.execute('''
                INSERT INTO messages 
                (conversation_id, model, role, content) 
                VALUES (?, ?, ?, ?)
            ''', (conv_id, model, role, content))
        
        conn.commit()
    except Exception as e:
        print(f"Error saving message: {e}")
    finally:
        conn.close()

def load_chat_history(model_name):
    """
    Load chat history for a specific model
    
    Args:
        model_name (str): Model name to filter messages
    
    Returns:
        list: List of message contents
    """
    try:
        conn = sqlite3.connect('conversations.db')
        c = conn.cursor()
        
        # Get conversation IDs for this model
        c.execute('''SELECT DISTINCT conversation_id FROM messages 
                     WHERE model = ?''', (model_name,))
        
        conversation_ids = c.fetchall()
        history = []
        
        for conv_id in conversation_ids:
            conv_id = conv_id[0]
            
            # Get messages for this conversation ordered by timestamp
            c.execute('''SELECT content, role FROM messages 
                         WHERE model = ? AND conversation_id = ? 
                         ORDER BY timestamp''', (model_name, conv_id))
            
            results = c.fetchall()
            
            # Process results to include thinking process for models that support it
            i = 0
            while i < len(results):
                content, role = results[i]
                
                if role == "user":
                    # For user messages, check what follows
                    user_content = content
                    
                    # Check if we have an assistant response
                    if i + 1 < len(results) and results[i + 1][1] == "assistant":
                        # Regular user -> assistant sequence
                        history.append((user_content, results[i + 1][0]))
                        i += 2
                    else:
                        # User message with no response yet
                        history.append((user_content, None))
                        i += 1
                else:
                    # Skip any other messages not handled above
                    i += 1
        
        conn.close()
        return history
    except Exception as e:
        print(f"Error loading chat history: {e}")
        traceback.print_exc()
        return []

def create_conversation(initial_message, model="llama3.1:8b"):
    """
    Create a new conversation
    
    Args:
        initial_message (str): First message in the conversation
        model (str): Model to use for the conversation
    
    Returns:
        int: Conversation ID
    """
    try:
        conn = sqlite3.connect('conversations.db')
        cursor = conn.cursor()
        
        # Special handling for image generation
        if model == "SDXL":
            # Generate a title based on the image prompt
            title = generate_conversation_title(initial_message, "", model)
        else:
            # Default title for other conversations
            title = "Untitled Conversation"
        
        # Create conversation with the generated title
        cursor.execute('''
            INSERT INTO conversations (title, last_used_model)
            VALUES (?, ?)
        ''', (title, model))
        
        conv_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return conv_id
        
    except Exception as e:
        print(f"Error creating conversation: {e}")
        traceback.print_exc()
        return None

def generate_conversation_title(message, response, model):
    """
    Generate a title for a conversation
    
    Args:
        message (str): User's message
        response (str): Assistant's response
        model (str): Model or bot name used for generation
    
    Returns:
        str: Generated title
    """
    try:
        # Get the current title if it exists
        if 'conv_id' in locals() or 'conv_id' in globals():
            conn = sqlite3.connect('conversations.db')
            c = conn.cursor()
            c.execute('SELECT title FROM conversations WHERE id = ?', (conv_id,))
            result = c.fetchone()
            conn.close()
            
            # If title already exists and is not "Untitled Conversation", return it
            if result and result[0] and result[0].strip() != "Untitled Conversation":
                return result[0]
        
        # Get the default model for title generation
        default_model = get_default_model()
        
        # Special handling for image generation conversations
        if model == "SDXL" or "image" in message.lower():
            import re
            
            # Clean and normalize the message
            clean_message = message.lower().strip()
            clean_message = re.sub(r'^(generate|create|make)\s*(an?\s*)?image\s*(of|with)?', '', clean_message).strip()
            clean_message = re.sub(r'--\w+\s*[^\s]+', '', clean_message).strip()
            
            # Prepare messages for title generation
            messages = [
                {
                    "role": "system",
                    "content": "Create a short, concise title (3-4 words) for this image prompt. Return ONLY the title."
                },
                {
                    "role": "user",
                    "content": f"Image prompt: {clean_message}"
                }
            ]
            
            title = ""
            for chunk in ollama_client.chat(
                model=get_model_version(default_model),
                messages=messages,
                stream=True,
                options={
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_predict": 30
                }
            ):
                if chunk['message']['content']:
                    content = chunk['message']['content']
                    if "<think>" in content:
                        content = content.split("<think>")[0].strip()
                    title += content
            
            # Clean up title
            title = title.strip().strip('"').strip().title()
            
            # Fallback and truncate
            if not title:
                words = clean_message.split()
                title = ' '.join(words[:3]).title()
            
            # Ensure title is not too long
            title = title[:30] if title else "Image Generation"
        else:
            # For all other conversation types
            system_prompt = "Create a very short, concise title (3-4 words) for this conversation. Return ONLY the title."
            
            # Use the default model to generate a title
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": f"User: {message}\nAssistant: {response}"
                }
            ]
            
            title = ""
            for chunk in ollama_client.chat(
                model=get_model_version(default_model),
                messages=messages,
                stream=True,
                options={
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_predict": 30
                }
            ):
                if chunk['message']['content']:
                    content = chunk['message']['content']
                    if "<think>" in content:
                        content = content.split("<think>")[0].strip()
                    title += content
            
            # Clean up title and ensure it's concise
            title = title.strip().strip('"').strip().title()
            
            # Limit to 6 words maximum
            words = title.split()
            if len(words) > 6:
                title = ' '.join(words[:6])
            
            # Fallback
            if not title:
                title = "Untitled Conversation"
        
        return title
    finally:
        # Update the conversation title in the database
        if 'conv_id' in locals() or 'conv_id' in globals():
            update_conversation_title(conv_id, title)

def update_conversation_title(conv_id, title):
    """
    Update the title of a conversation
    
    Args:
        conv_id (int): Conversation ID
        title (str): New title
    """
    try:
        conn = sqlite3.connect('conversations.db')
        c = conn.cursor()
        
        # Update conversation title
        c.execute('''UPDATE conversations 
                     SET title = ? 
                     WHERE id = ?''', (title.strip(), conv_id))
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error updating conversation title: {e}")
def delete_conversation(conv_id):
    """
    Delete a specific conversation and its associated messages
    
    Args:
        conv_id (str): Conversation ID to delete
    
    Returns:
        tuple: Updated conversation list, empty chatbot, empty text, None state, default model
    """
    try:
        # Delete conversation and its messages
        conn = sqlite3.connect('conversations.db')
        c = conn.cursor()
        
        # Delete messages associated with the conversation
        c.execute('DELETE FROM messages WHERE conversation_id = ?', (conv_id,))
        
        # Delete the conversation
        c.execute('DELETE FROM conversations WHERE id = ?', (conv_id,))
        
        conn.commit()
        conn.close()
        
        # Retrieve updated conversation list
        conversations = get_conversation_list()
        
        # Combine all available models
        available_models = [get_model_display_name(m) for m in get_available_models()]  # Ollama models
        bot_names = get_bot_names()  # Saved bot names
        all_models = available_models + bot_names + ["SDXL"]  # Combine all models
        
        # Return all expected outputs
        return (
            pd.DataFrame(
                [[conv['title']] for conv in conversations],
                columns=['Title']
            ),  # Conversation list DataFrame
            [],  # Empty chatbot
            "",  # Empty text
            None,  # State (None)
            gr.Dropdown(
                choices=all_models, 
                value=get_default_model()
            )  # Model dropdown reset
        )
    
    except Exception as e:
        print(f"Error deleting conversation: {e}")
        
        # Return default/empty values in case of error
        return (
            pd.DataFrame(columns=['Title']),
            [],
            f"Error deleting conversation: {str(e)}",
            None,
            gr.Dropdown(
                choices=all_models, 
                value=get_default_model()
            )
        )

def get_conversations():
    """
    Retrieve conversations for Gradio DataFrame display
    
    Args:
        None
    
    Returns:
        pandas.DataFrame: DataFrame with only title column
    """
    try:
        # Get conversation list
        conversations = get_conversation_list()
        
        # Convert to DataFrame with only title column
        if not conversations:
            return pd.DataFrame(columns=['Title'])
        
        return pd.DataFrame(
            [[conv['title']] for conv in conversations],
            columns=['Title']
        )
    
    except Exception as e:
        print(f"Error formatting conversation list: {e}")
        return pd.DataFrame(columns=['Title'])

def get_conversation_list():
    """
    Retrieve a list of all conversations with detailed information
    
    Args:
        None
    
    Returns:
        list: List of dictionaries containing conversation details
    """
    try:
        conn = sqlite3.connect('conversations.db')
        c = conn.cursor()
        
        # Fetch conversations with their last used model
        c.execute('''SELECT 
                        id, 
                        title, 
                        last_used_model, 
                        created_at 
                     FROM conversations 
                     ORDER BY created_at DESC''')
        
        # Columns to match the DataFrame in delete_conversation
        columns = ['id', 'title', 'last_used_model', 'created_at']
        
        # Convert results to list of dictionaries
        conversations = [dict(zip(columns, row)) for row in c.fetchall()]
        
        # Check and replace base models with bot names if applicable
        for conv in conversations:
            bot_config = get_bot_by_name(conv['last_used_model'])
            if bot_config:
                conv['last_used_model'] = bot_config['name']
        
        conn.close()
        return conversations
    
    except Exception as e:
        print(f"Error retrieving conversation list: {e}")
        return []

def format_conversation_list(search_query=None):
    """
    Format conversations into a DataFrame for display in the UI
    
    Args:
        search_query (str, optional): Filter conversations by search query
    
    Returns:
        pandas.DataFrame: Formatted conversation list
    """
    try:
        # Fetch conversations
        conversations = get_conversation_list()
        
        # Apply search filter if provided
        if search_query:
            conversations = [
                conv for conv in conversations 
                if search_query.lower() in conv['title'].lower()
            ]
        
        # Create DataFrame with only title column
        df = pd.DataFrame(conversations)[['title']] if conversations else pd.DataFrame({'title': []})
        
        # Reset index to ensure proper row selection
        df = df.reset_index(drop=True)
        
        # Ensure proper column types
        df = df.astype({'title': 'str'})
        
        print(f"Formatted conversation list: {df}")
        return df
    
    except Exception as e:
        print(f"Error formatting conversation list: {e}")
        traceback.print_exc()
        return pd.DataFrame({'title': []})

def handle_conversation_click(evt: gr.SelectData, conv_list):
    """
    Handle click event on conversation list
    
    Args:
        evt (gr.SelectData): Gradio selection event
        conv_list (pandas.DataFrame): List of conversations
    
    Returns:
        tuple: Conversation ID, chat history, model, conversation title
    """
    try:
        print(f"Event index: {evt.index}")
        print(f"Conversation list: {conv_list}")
        
        # Get the title of the selected conversation
        selected_title = conv_list.iloc[evt.index[0]][0]
        
        # Fetch conversations to get the corresponding ID
        conn = sqlite3.connect('conversations.db')
        c = conn.cursor()
        
        # Directly query for the conversation ID by title
        c.execute('''SELECT id FROM conversations 
                     WHERE title = ? 
                     ORDER BY created_at DESC 
                     LIMIT 1''', (selected_title,))
        
        conv_result = c.fetchone()
        conn.close()
        
        if not conv_result:
            print(f"No conversation found with title: {selected_title}")
            return None, [], get_model_display_name("llama3.1:8b"), "New Conversation"
        
        # Get the conversation ID
        conv_id = conv_result[0]
        
        # Load conversation details
        title, model, chat_history = load_conversation(conv_id)
        
        # Format the model name for display
        if model == "SDXL":
            # Keep SDXL model as is
            display_model = "SDXL"
        elif get_bot_by_name(model):
            # Keep bot names as is
            display_model = model
        else:
            # Format Ollama model names
            display_model = get_model_display_name(model or "llama3.1:8b")
        
        print(f"Loaded conversation with model: {model}, display model: {display_model}")
        
        return (
            conv_id,  # Current conversation ID
            chat_history,  # Loaded chat history
            display_model,  # Properly formatted model name
            title  # Conversation title
        )
    
    except Exception as e:
        print(f"Error handling conversation click: {e}")
        traceback.print_exc()
        return None, [], get_model_display_name("llama3.1:8b"), "New Conversation"

def load_conversation(conv_id):
    """
    Load a specific conversation, ensuring bot configurations are properly handled
    
    Args:
        conv_id (str): Conversation ID
    
    Returns:
        tuple: Conversation title, model/bot name, chat history
    """
    try:
        # Get conversation details
        title, model, messages = get_conversation_details(conv_id)
        
        if not messages:
            return None, None, []
        
        # Process messages into chat history format
        chat_history = []
        current_user_msg = None
        current_assistant_msg = None
        has_generated_image = False
        
        for msg in messages:
            if msg['role'] == 'user':
                # If we have a complete message pair, add it to history
                if current_user_msg is not None and current_assistant_msg is not None:
                    chat_history.append((current_user_msg, current_assistant_msg))
                current_user_msg = msg['content']
                current_assistant_msg = None
            
            elif msg['role'] == 'assistant':
                content = msg['content']
                # Check if this is an image message
                if '[Generated Image]' in content or content.startswith('![Generated Image]('):
                    has_generated_image = True
                    # Extract the image path and ensure it uses the correct format
                    import re
                    match = re.search(r'!\[Generated Image\]\((file/|gradio_api/file=)?(.+)\)', content)
                    if match:
                        image_path = match.group(2)
                        # Remove any 'file/' or 'gradio_api/file=' prefix if it exists
                        image_path = re.sub(r'^(file/|gradio_api/file=)', '', image_path)
                        # Ensure the path starts with '/gradio_api/file='
                        if not image_path.startswith('/gradio_api/file='):
                            image_path = f'/gradio_api/file={image_path}'
                        content = f"![Generated Image]({image_path})"
                current_assistant_msg = content
        
        # Add the last message pair if it exists
        if current_user_msg and current_assistant_msg:
            chat_history.append((current_user_msg, current_assistant_msg))
        
        # If we detected generated images, force the model to SDXL
        if has_generated_image:
            model = "SDXL"
            # Update the model in the database to maintain consistency
            update_conversation_model(model, conv_id)
        
        return title, model, chat_history
    
    except Exception as e:
        print(f"Error loading conversation: {e}")
        traceback.print_exc()
        return None, None, []

def get_conversation_details(conv_id):
    """
    Retrieve conversation details, preserving bot name if used
    
    Args:
        conv_id (str): Conversation ID
    
    Returns:
        tuple: (title, last_used_model, messages)
    """
    try:
        conn = sqlite3.connect('conversations.db')
        c = conn.cursor()
        
        # Fetch conversation details
        c.execute('''SELECT 
                        title, 
                        last_used_model, 
                        created_at 
                     FROM conversations 
                     WHERE id = ?''', (conv_id,))
        conv_details = c.fetchone()
        
        if not conv_details:
            conn.close()
            return None, None, []
        
        title, last_used_model, _ = conv_details
        
        # Check if the last used model is a bot
        bot_config = get_bot_by_name(last_used_model)
        if bot_config:
            # If it was a bot, use the bot name
            last_used_model = bot_config['name']
        
        # Fetch messages for the conversation
        c.execute('''SELECT 
                        role, 
                        content, 
                        timestamp 
                     FROM messages 
                     WHERE conversation_id = ? 
                     ORDER BY timestamp''', (conv_id,))
        
        # Collect messages and check for generated images
        messages = []
        has_generated_image = False
        for row in c.fetchall():
            role, content, timestamp = row
            
            # Check if content contains image markdown
            if content.startswith('![Generated Image](') or '[Generated Image]' in content:
                has_generated_image = True
                # Extract the image path
                import re
                match = re.search(r'!\[Generated Image\]\((file/|gradio_api/file=)?(.+)\)', content)
                if match:
                    image_path = match.group(2)
                    # Remove any 'file/' or 'gradio_api/file=' prefix if it exists
                    image_path = re.sub(r'^(file/|gradio_api/file=)', '', image_path)
                    # Ensure the path starts with 'gradio_api/file='
                    if not image_path.startswith('gradio_api/file='):
                        image_path = f'gradio_api/file={image_path}'
                    content = f"![Generated Image]({image_path})"
            
            messages.append({
                'role': role,
                'content': content,
                'timestamp': timestamp
            })
        
        # If we detected generated images but the model isn't SDXL or a workflow,
        # update it to SDXL (for backward compatibility)
        if has_generated_image and not (last_used_model == "SDXL" or last_used_model.startswith("Workflow: ")):
            last_used_model = "SDXL"
            # Update the model in the database
            c.execute('UPDATE conversations SET last_used_model = ? WHERE id = ?', ("SDXL", conv_id))
            conn.commit()
        
        conn.close()
        return title, last_used_model, messages
    
    except Exception as e:
        print(f"Error getting conversation details: {e}")
        traceback.print_exc()
        return None, None, []

def rename_current_chat(conv_id, new_title):
    """
    Rename the current conversation
    
    Args:
        conv_id (int): ID of the conversation to rename
        new_title (str): New title for the conversation
    
    Returns:
        str: Updated conversation title
    """
    if not conv_id or not new_title:
        return "Untitled Conversation"
    
    try:
        conn = sqlite3.connect('conversations.db')
        c = conn.cursor()
        
        # Update conversation title
        c.execute('''UPDATE conversations 
                     SET title = ? 
                     WHERE id = ?''', (new_title.strip(), conv_id))
        
        conn.commit()
        conn.close()
        
        return new_title.strip()
    
    except Exception as e:
        print(f"Error renaming conversation: {e}")
        traceback.print_exc()
        return "Untitled Conversation"

def search_conversations(query):
    conn = sqlite3.connect('conversations.db')
    c = conn.cursor()
    c.execute('''SELECT id, title, created_at FROM conversations
                WHERE title LIKE ? OR id IN
                    (SELECT conversation_id FROM messages WHERE content LIKE ?)
                ORDER BY created_at DESC''',
             (f'%{query}%', f'%{query}%'))
    results = c.fetchall()
    conn.close()
    return [[row[0], row[1], row[2]] for row in results]

def update_chat_info(conv_id):
    title = get_chat_title(conv_id)
    return title

def get_chat_title(conv_id):
    if not conv_id:
        return "New Chat"
    conn = sqlite3.connect('conversations.db')
    c = conn.cursor()
    c.execute('SELECT title FROM conversations WHERE id = ?', (conv_id,))
    result = c.fetchone()
    conn.close()
    return result[0] if result else "New Chat"

def update_conversation_model(model, conv_id):
    """
    Update the model used for a conversation
    
    Args:
        model (str): Model to use for the conversation
        conv_id (str): Conversation ID
        
    Returns:
        str: Updated model name
    """
    if conv_id:
        try:
            # Check if this is an image generation conversation
            conn = sqlite3.connect('conversations.db')
            c = conn.cursor()
            
            # Check messages for image generation
            c.execute('''SELECT content FROM messages 
                        WHERE conversation_id = ? 
                        AND (content LIKE '%![Generated Image]%' 
                             OR content LIKE '%[Generated Image]%')''', (conv_id,))
            has_images = bool(c.fetchone())
            
            # If this is an image generation conversation and the new model is not SDXL or a workflow,
            # only then force SDXL (for backward compatibility)
            if has_images and not (model == "SDXL" or model.startswith("Workflow: ")):
                model = "SDXL"
            
            # Update the model in the database
            c.execute('''UPDATE conversations 
                        SET last_used_model = ? 
                        WHERE id = ?''', (model, conv_id))
            conn.commit()
            conn.close()
            print(f"Updated conversation {conv_id} model to {model}")
        except Exception as e:
            print(f"Error updating conversation model: {e}")
            traceback.print_exc()
    return model

def get_default_model():
    """
    Retrieve the default fallback model
    
    Args:
        None
    
    Returns:
        str: Name of the default model
    """
    try:
        conn = sqlite3.connect('conversations.db')
        c = conn.cursor()
        
        # Check if default model is set
        c.execute('SELECT value FROM settings WHERE key = "default_model"')
        result = c.fetchone()
        
        conn.close()
        
        if result:
            model_name = result[0]
            # If it's an Ollama model, get the display name
            if model_name == "SDXL":
                return "SDXL"
            elif get_bot_by_name(model_name):
                return model_name
            else:
                return get_model_display_name(model_name)
        
        # Return formatted default model name
        return get_model_display_name("llama3.1:8b")
    except Exception as e:
        print(f"Error retrieving default model: {e}")
        traceback.print_exc()
        return get_model_display_name("llama3.1:8b")

def set_default_model(model):
    """
    Set the default fallback model
    
    Args:
        model (str): Name of the model to set as default
    
    Returns:
        str: The set default model
    """
    try:
        conn = sqlite3.connect('conversations.db')
        c = conn.cursor()
        
        # Create settings table if not exists
        c.execute('''CREATE TABLE IF NOT EXISTS settings 
                     (key TEXT PRIMARY KEY, value TEXT)''')
        
        # Insert or replace default model
        c.execute('''INSERT OR REPLACE INTO settings (key, value) 
                     VALUES ('default_model', ?)''', (model,))
        
        conn.commit()
        conn.close()
        
        return model
    except Exception as e:
        print(f"Error setting default model: {e}")
        return model

def load_ollama_models():
    """
    Load and return available Ollama models for the dialog
    
    Args:
        None
    
    Returns:
        pandas.DataFrame: DataFrame with available models
    """
    try:
        models = get_available_ollama_models()
        
        # Add SDXL as a special model option
        models.append({
            'Model Name': 'SDXL', 
            'Size': 'N/A', 
            'Actions': 'N/A'
        })
        
        # Convert to DataFrame
        df = pd.DataFrame(models)
        return df
    except Exception as e:
        print(f"Error loading Ollama models: {e}")
        return pd.DataFrame(columns=['Model Name', 'Size', 'Actions'])

def open_manage_bots_dialog():
    """
    Open the manage bots dialog and prepare its contents
    
    Args:
        None
    
    Returns:
        tuple: Dialog visibility, existing bots list, bot name dropdown, base model dropdown, save status
    """
    try:
        # Retrieve existing bots
        existing_bots = get_existing_bots_dataframe()
        
        # Get available models for base model selection
        available_models = get_available_models()
        
        # Get existing bot names
        existing_bot_names = get_existing_bots()
        
        return (
            gr.update(visible=True),  # Dialog visibility
            existing_bots,            # Existing bots list
            gr.Dropdown(choices=existing_bot_names + ['New Bot'], label="Select or Create Bot"),  # Bot name dropdown
            gr.Dropdown(choices=available_models, label="Base Model"),  # Base model dropdown
            ""  # Clear save status
        )
    
    except Exception as e:
        print(f"Error opening manage bots dialog: {e}")
        return (
            gr.update(visible=True),  # Dialog visibility
            pd.DataFrame(columns=['Name', 'Base Model', 'System Prompt', 'Temperature', 'Max Tokens', 'Top P']),
            gr.Dropdown(choices=['New Bot'], label="Select or Create Bot"),
            gr.Dropdown(choices=get_available_models(), label="Base Model"),
            f"Error: {str(e)}"
        )

# Modify the manage bots button click event
def handle_bot_save(bot_name_dropdown, bot_name_input, base_model, system_prompt, temperature, max_tokens, top_p, model_dropdown):
    """
    Save or update bot configuration
    
    Args:
        bot_name_dropdown (str): Selected bot name from dropdown
        bot_name_input (str): Custom bot name input
        base_model (str): Base model for the bot
        system_prompt (str): System prompt for the bot
        temperature (float): Temperature setting
        max_tokens (int): Maximum tokens
        top_p (float): Top P sampling
        model_dropdown (gr.Dropdown): Model dropdown component
    
    Returns:
        tuple: Updated bots list, bot name dropdown, base model dropdown, save status, model dropdown update
    """
    try:
        # Determine bot name
        if bot_name_dropdown == 'New Bot':
            bot_name = bot_name_input.strip()
        else:
            bot_name = bot_name_dropdown
        
        # Validate inputs
        if not bot_name:
            return (
                get_existing_bots_dataframe(), 
                gr.update(choices=get_existing_bots() + ['New Bot']), 
                gr.update(choices=get_available_models()), 
                "Bot name cannot be empty",
                gr.update(choices=get_available_models())
            )
            
        if not system_prompt:
            return (
                get_existing_bots_dataframe(),
                gr.update(choices=get_existing_bots() + ['New Bot']),
                gr.update(choices=get_available_models()),
                "System prompt cannot be empty",
                gr.update(choices=get_available_models())
            )
            
        # Validate numeric parameters
        try:
            temperature = float(temperature)
            max_tokens = int(max_tokens)
            top_p = float(top_p)
            
            if not (0.0 <= temperature <= 2.0):
                raise ValueError("Temperature must be between 0.0 and 2.0")
            if not (0.0 <= top_p <= 1.0):
                raise ValueError("Top P must be between 0.0 and 1.0")
            if max_tokens <= 0:
                raise ValueError("Max tokens must be greater than 0")
                
        except (ValueError, TypeError) as e:
            return (
                get_existing_bots_dataframe(),
                gr.update(choices=get_existing_bots() + ['New Bot']),
                gr.update(choices=get_available_models()),
                f"Invalid parameter values: {str(e)}",
                gr.update(choices=get_available_models())
            )
        
        # Save bot configuration
        success, message = save_bot(
            bot_name, 
            base_model, 
            system_prompt, 
            temperature, 
            max_tokens, 
            top_p
        )
        
        # Get updated list of existing bots and models
        updated_bots = get_existing_bots()
        updated_models = get_available_models()
        
        # Return updated components
        return (
            get_existing_bots_dataframe(),  # Updated bots list
            gr.update(choices=updated_bots + ['New Bot']),  # Bot name dropdown update
            gr.update(choices=updated_models),  # Base model dropdown update
            message,  # Save status message
            gr.update(choices=updated_models)  # Model dropdown update
        )
    
    except Exception as e:
        print(f"Error saving bot configuration: {e}")
        return (
            get_existing_bots_dataframe(), 
            gr.update(choices=get_existing_bots() + ['New Bot']), 
            gr.update(choices=get_available_models()), 
            f"Error: {str(e)}",
            gr.update(choices=get_available_models())
        )

def get_existing_bots_dataframe():
    """
    Retrieve existing bots as a DataFrame
    
    Args:
        None
    
    Returns:
        pd.DataFrame: DataFrame containing bot configurations
    """
    try:
        conn = sqlite3.connect('bots.db')
        df = pd.read_sql_query('''
            SELECT 
                name AS Name, 
                base_model AS "Base Model", 
                system_prompt AS "System Prompt", 
                temperature AS Temperature, 
                max_tokens AS "Max Tokens", 
                top_p AS "Top P" 
            FROM bots
        ''', conn)
        conn.close()
        return df
    
    except Exception as e:
        print(f"Error retrieving bots DataFrame: {e}")
        return pd.DataFrame(columns=['Name', 'Base Model', 'System Prompt', 'Temperature', 'Max Tokens', 'Top P'])

def get_existing_bots():
    """
    Retrieve list of existing bot names
    
    Args:
        None
    
    Returns:
        list: Names of existing bots
    """
    try:
        conn = sqlite3.connect('bots.db')
        c = conn.cursor()
        
        # Fetch all bot names
        c.execute('SELECT name FROM bots')
        bots = [row[0] for row in c.fetchall()]
        
        conn.close()
        return bots
    
    except Exception as e:
        print(f"Error retrieving existing bots: {e}")
        return []

def load_bot_configuration(bot_name):
    """
    Load configuration for a specific bot
    
    Args:
        bot_name (str): Name of the bot to load
    
    Returns:
        dict: Bot configuration details with all parameters
    """
    try:
        conn = sqlite3.connect('bots.db')
        cursor = conn.cursor()
        
        # Fetch bot configuration
        cursor.execute('''
            SELECT 
                name, 
                base_model, 
                system_prompt, 
                temperature, 
                max_tokens, 
                top_p 
            FROM bots 
            WHERE name = ?
        ''', (bot_name,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'name': result[0],
                'base_model': result[1],
                'system_prompt': result[2],
                'temperature': result[3],
                'max_tokens': result[4],
                'top_p': result[5]
            }
        else:
            return None
    
    except Exception as e:
        print(f"Error loading bot configuration for {bot_name}: {e}")
        return None

def delete_bot_configuration(bot_name_dropdown):
    """
    Delete a bot configuration from the database
    
    Args:
        bot_name_dropdown (str): Name of the bot to delete
    
    Returns:
        tuple: Updated bots list, bot name dropdown, save status
    """
    try:
        # Prevent deleting 'New Bot'
        if not bot_name_dropdown or bot_name_dropdown == 'New Bot':
            return (
                get_existing_bots_dataframe(),
                gr.update(choices=get_existing_bots() + ['New Bot'], value='New Bot'),
                "Please select a valid bot to delete"
            )
        
        # Delete bot from database
        conn = sqlite3.connect('bots.db')
        cursor = conn.cursor()
        
        # Delete the bot configuration
        cursor.execute('DELETE FROM bots WHERE name = ?', (bot_name_dropdown,))
        conn.commit()
        conn.close()
        
        # Get updated list of bots
        updated_bots = get_existing_bots()
        
        return (
            get_existing_bots_dataframe(),  # Updated bots list
            gr.update(choices=updated_bots + ['New Bot'], value='New Bot'),  # Bot name dropdown update
            f"Bot '{bot_name_dropdown}' deleted successfully"  # Status message
        )
    
    except Exception as e:
        print(f"Error deleting bot configuration: {e}")
        return (
            get_existing_bots_dataframe(),
            gr.update(choices=get_existing_bots() + ['New Bot'], value='New Bot'),
            f"Error deleting bot: {str(e)}"
        )

def download_ollama_model(model_name: str, progress=gr.Progress()):
    try:
        if not model_name:
            return "Please enter a valid model name"
        
        progress(0, desc="Initializing download...")
        response = requests.post(
            'http://localhost:11434/api/pull',
            json={'name': model_name},
            stream=True
        )
        response.raise_for_status()

        # Simulate progress for models without content-length
        total_size = int(response.headers.get('content-length', 1024 * 1024))  # Default to 1MB
        downloaded = 0
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                downloaded += len(chunk)
                progress(min(downloaded / total_size, 1.0), desc=f"Downloading {model_name}...")
        
        return f"Successfully downloaded {model_name}"
    except requests.exceptions.RequestException as e:
        return f"Network error downloading model: {str(e)}"
    except Exception as e:
        return f"Error downloading model: {str(e)}"

def process_document(file_path, original_name, conversation_id=None):
    """
    Process an uploaded document for RAG
    
    Args:
        file_path (str): Path to the temporary uploaded file
        original_name (str): Original filename
        conversation_id (int, optional): Associated conversation ID
    
    Returns:
        tuple: (success, message)
    """
    try:
        print(f"\nProcessing document: {original_name}")
        print(f"Conversation ID: {conversation_id}")
        print(f"File path: {file_path}")
        
        # Create a unique filename to avoid collisions
        unique_filename = f"{str(uuid.uuid4())}_{original_name}"
        file_ext = os.path.splitext(original_name)[1].lower()
        
        # Create docs directory if it doesn't exist
        docs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")
        os.makedirs(docs_dir, exist_ok=True)
        
        # Copy file to permanent location
        permanent_path = os.path.join(docs_dir, unique_filename)
        shutil.copy2(file_path, permanent_path)
        print(f"File copied to: {permanent_path}")
        
        # Select appropriate loader based on file type
        if file_ext == '.pdf':
            loader = PyPDFLoader(permanent_path)
            print("Using PDF loader")
        elif file_ext == '.txt':
            loader = TextLoader(permanent_path)
            print("Using text loader")
        elif file_ext == '.doc':
            loader = Docx2txtLoader(permanent_path)
            print("Using Word document loader")
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        # Load and split the document
        print("Loading document...")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            add_start_index=True,
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Document split into {len(chunks)} chunks")
        
        # Create a new embeddings directory with a unique ID
        embeddings_dir = os.path.join(docs_dir, "embeddings", str(uuid.uuid4()))
        
        # Delete the directory if it already exists (should not happen with UUID)
        if os.path.exists(embeddings_dir):
            shutil.rmtree(embeddings_dir)
            
        # Create the directory
        os.makedirs(embeddings_dir, exist_ok=True)
        print(f"Created embeddings directory: {embeddings_dir}")
        
        # Initialize embeddings model using TfidfEmbeddings
        print("Initializing TF-IDF embeddings model...")
        # Extract text from chunks for fitting the TF-IDF vectorizer
        chunk_texts = [chunk.page_content for chunk in chunks]
        embeddings = TfidfEmbeddings()
        
        # Generate unique IDs for each chunk
        ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
        
        # Create and persist vector store
        print("Creating vector store...")
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=embeddings_dir,
            ids=ids
        )
        vectordb.persist()
        print("Vector store created and persisted")
        
        # Save document info to database
        print("Saving document info to database...")
        conn = sqlite3.connect('conversations.db')
        c = conn.cursor()
        
        # Add metadata to chunks
        for chunk in chunks:
            if 'source' not in chunk.metadata:
                chunk.metadata['source'] = original_name
        
        # Convert conversation_id to integer if it's a string
        if conversation_id and isinstance(conversation_id, str):
            try:
                conversation_id = int(conversation_id)
                print(f"Converted conversation_id string to integer: {conversation_id}")
            except ValueError:
                print(f"Warning: Invalid conversation_id format: {conversation_id}")
                conversation_id = None
        
        c.execute('''INSERT INTO documents 
                     (filename, original_name, file_type, embedding_path, conversation_id)
                     VALUES (?, ?, ?, ?, ?)''',
                  (unique_filename, original_name, file_ext, embeddings_dir, conversation_id))
        
        conn.commit()
        print(f"Document saved to database with conversation_id: {conversation_id}")
        
        # Verify the document was saved
        c.execute('SELECT id FROM documents WHERE filename = ? AND conversation_id = ?', 
                 (unique_filename, conversation_id))
        doc_id = c.fetchone()
        print(f"Verified document saved with ID: {doc_id[0] if doc_id else None}")
        
        conn.close()
        
        return True, f"Successfully processed {original_name}"
        
    except Exception as e:
        print(f"Error processing document: {e}")
        traceback.print_exc()
        return False, f"Error processing document: {str(e)}"

def handle_file_upload(file_obj, conversation_id=None):
    """Handle file upload from Gradio interface
    
    Args:
        file_obj: Gradio file object
        conversation_id (int, optional): Current conversation ID
    
    Returns:
        tuple: (status_message, document_name, overlay_visibility_state, conversation_id)
    """
    try:
        print(f"\nHandling file upload for conversation ID: {conversation_id}")
        
        if file_obj is None:
            print("No file object provided")
            return "", "", False, conversation_id
            
        original_name = os.path.basename(file_obj)
        print(f"Processing file: {original_name}")
        
        # Create new conversation if none exists
        if not conversation_id:
            conversation_id = create_conversation("New conversation with document: " + original_name)
            if not conversation_id:
                print("Failed to create new conversation")
                return "Error: Could not create conversation", "", False, None
            print(f"Created new conversation with ID: {conversation_id}")
        
        # Verify conversation exists
        conn = sqlite3.connect('conversations.db')
        c = conn.cursor()
        
        # Check if the conversation exists by ID if ID is provided
        if conversation_id:
            c.execute('SELECT id FROM conversations WHERE id = ?', (conversation_id,))
            conv_result = c.fetchone()
        else:
            # This case should not happen as we create a conversation if none exists above
            c.execute('''SELECT id FROM conversations 
                         WHERE title = ? 
                         ORDER BY created_at DESC 
                         LIMIT 1''', ("New conversation with document: " + original_name,))
            conv_result = c.fetchone()
        
        conn.close()
        
        if not conv_result:
            print(f"No conversation found with ID: {conversation_id}")
            return f"Error: Invalid conversation ID", "", False, None
        print(f"Verified conversation {conversation_id} exists")
        
        success, message = process_document(file_obj, original_name, conversation_id)
        print(f"Document processing result: {success}, {message}")
        
        if success:
            return message, f"**{original_name}**", True, conversation_id
        else:
            return message, "", False, conversation_id
        
    except Exception as e:
        print(f"Error handling file upload: {e}")
        traceback.print_exc()
        return f"Error uploading file: {str(e)}", "", False, conversation_id

def get_relevant_context(query, conversation_id=None, top_k=3):
    """
    Get relevant context from uploaded documents
    
    Args:
        query (str): User's query
        conversation_id (int, optional): Current conversation ID
        top_k (int): Number of chunks to retrieve
    
    Returns:
        str: Relevant context from documents
    """
    try:
        print(f"\nGetting context for query: {query}")
        print(f"Conversation ID: {conversation_id}")
        
        # Get all documents for the conversation
        conn = sqlite3.connect('conversations.db')
        c = conn.cursor()
        
        if conversation_id:
            c.execute('SELECT embedding_path FROM documents WHERE conversation_id = ? AND deleted = 0', (conversation_id,))
        else:
            c.execute('SELECT embedding_path FROM documents WHERE conversation_id IS NULL AND deleted = 0')
            
        embedding_paths = c.fetchall()
        print(f"Found {len(embedding_paths)} document(s)")
        conn.close()
        
        if not embedding_paths:
            print("No documents found")
            return ""
        
        # Initialize TF-IDF embeddings model
        print("Initializing TF-IDF embeddings model...")
        embeddings = TfidfEmbeddings()
        
        # Combine results from all documents
        all_results = []
        for path in embedding_paths:
            print(f"Processing embeddings from: {path[0]}")
            if not os.path.exists(path[0]):
                print(f"Warning: Embedding path does not exist: {path[0]}")
                continue
                
            try:
                # Use Chroma with TF-IDF embeddings
                vectordb = Chroma(
                    persist_directory=path[0],
                    embedding_function=embeddings
                )
                results = vectordb.similarity_search(query, k=top_k)
                print(f"Found {len(results)} relevant chunks using TF-IDF search")
                all_results.extend(results)
            except Exception as e:
                print(f"Error using TF-IDF search: {e}")
                
                # Fallback to keyword-based search if TF-IDF fails
                print("Falling back to keyword-based search")
                
                # Try to load the collection files directly
                try:
                    # Look for document chunks in the collection directory
                    collection_dir = os.path.join(path[0], "chroma-embeddings.parquet")
                    if os.path.exists(collection_dir):
                        import pandas as pd
                        # Try to load the parquet file with document contents
                        df = pd.read_parquet(collection_dir)
                        if 'document' in df.columns:
                            # Simple keyword matching
                            query_terms = query.lower().split()
                            for _, row in df.iterrows():
                                content = row['document'].lower()
                                # Calculate a simple relevance score based on term frequency
                                score = sum(1 for term in query_terms if term in content)
                                if score > 0:
                                    # Create a Document object with the content and metadata
                                    doc = Document(
                                        page_content=row['document'],
                                        metadata={'score': score, 'source': 'Unknown document'}
                                    )
                                    all_results.append(doc)
                            print(f"Found {len(all_results)} chunks using keyword search")
                    else:
                        # Try to find JSON files in the embedding directory
                        json_files = glob.glob(os.path.join(path[0], "chunk_*.json"))
                        if json_files:
                            print(f"Found {len(json_files)} JSON chunk files")
                            
                            # Load all chunks from JSON files
                            for json_file in json_files:
                                try:
                                    with open(json_file, 'r', encoding='utf-8') as f:
                                        chunk_data = json.load(f)
                                        content = chunk_data.get('page_content', '')
                                        # Simple keyword matching
                                        query_terms = query.lower().split()
                                        score = sum(1 for term in query_terms if term in content.lower())
                                        if score > 0:
                                            # Create a Document object with the content and metadata
                                            doc = Document(
                                                page_content=content,
                                                metadata={
                                                    'score': score,
                                                    'source': chunk_data.get('source', 'Unknown document')
                                                }
                                            )
                                            all_results.append(doc)
                                except Exception as json_error:
                                    print(f"Error loading chunk from {json_file}: {json_error}")
                except Exception as fallback_error:
                    print(f"Error in fallback search: {fallback_error}")
        
        # Sort by relevance and get top results
        if all_results:  # Check if list is not empty before accessing elements
            all_results = sorted(all_results, key=lambda x: x.metadata.get('score', 0), reverse=True)[:top_k]
        else:
            print("No results found from any document")
            return ""
        
        # Format context
        if all_results:
            formatted_chunks = []
            for i, doc in enumerate(all_results, 1):
                chunk = doc.page_content.strip()
                source = doc.metadata.get('source', 'Unknown document')
                print(f"Using chunk {i} from {source}")
                # Format each chunk with its source
                formatted_chunks.append(f"[Document {i} - Source: {source}]\n{chunk}")
            
            context = "\n\n".join(formatted_chunks)
            print("Successfully retrieved context")
            return context
        
        print("No relevant chunks found")
        return ""
        
    except Exception as e:
        print(f"Error getting relevant context: {e}")
        traceback.print_exc()
        return ""

def toggle_custom_bot_name_input(bot_name_dropdown):
    """
    Toggle visibility of custom bot name input based on dropdown selection
    
    Args:
        bot_name_dropdown (str): Selected bot name from dropdown
    
    Returns:
        tuple: Visibility of custom bot name input
    """
    return gr.update(visible=bot_name_dropdown == 'New Bot')

def process_bot_document(file_path, original_name, bot_id):
    """
    Process an uploaded document for a bot's knowledge base
    
    Args:
        file_path (str): Path to the temporary uploaded file
        original_name (str): Original filename
        bot_id (int): ID of the bot
    
    Returns:
        tuple: (success, message)
    """
    try:
        print(f"\nProcessing document for bot: {original_name}")
        print(f"Bot ID: {bot_id}")
        
        # Generate unique filename
        file_ext = os.path.splitext(original_name)[1].lower()
        unique_filename = f"{str(uuid.uuid4())}{file_ext}"
        
        # Create documents directory if it doesn't exist
        docs_dir = os.path.join('bot_documents')
        os.makedirs(docs_dir, exist_ok=True)
        
        # Copy file to permanent location
        permanent_path = os.path.join(docs_dir, unique_filename)
        shutil.copy2(file_path, permanent_path)
        print(f"File copied to: {permanent_path}")
        
        # Load and process document based on file type
        if file_ext == '.pdf':
            loader = PyPDFLoader(permanent_path)
            print("Using PDF loader")
        elif file_ext == '.docx':
            loader = Docx2txtLoader(permanent_path)
            print("Using Word document loader")
        elif file_ext == '.txt':
            loader = TextLoader(permanent_path)
            print("Using text loader")
        else:
            os.remove(permanent_path)
            return False, f"Unsupported file type: {file_ext}"
        
        # Load document and split into chunks
        print("Loading document...")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Document split into {len(chunks)} chunks")
        
        # Generate embeddings
        embeddings_dir = os.path.join('bot_embeddings')
        os.makedirs(embeddings_dir, exist_ok=True)
        
        # Create a unique embedding path with UUID to avoid conflicts
        unique_id = str(uuid.uuid4())
        embedding_path = os.path.join(embeddings_dir, unique_id)
        
        # Ensure the directory is fresh
        if os.path.exists(embedding_path):
            shutil.rmtree(embedding_path)
        os.makedirs(embedding_path, exist_ok=True)
        
        # Add source metadata to chunks
        for chunk in chunks:
            if 'source' not in chunk.metadata:
                chunk.metadata['source'] = original_name
        
        # Use TfidfEmbeddings instead of HuggingFaceEmbeddings
        print("Initializing TF-IDF embeddings model...")
        embedding_function = TfidfEmbeddings()
        
        # Generate unique IDs for each chunk
        ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
        
        try:
            # Create and persist vector store
            print("Creating vector store...")
            vectordb = Chroma.from_documents(
                documents=chunks,
                embedding=embedding_function,
                persist_directory=embedding_path,
                ids=ids
            )
            vectordb.persist()
            print("Vector store created and persisted")
        except Exception as e:
            print(f"Error creating vector store: {e}")
            print("Falling back to saving chunks as JSON files")
            
            # Save chunks as JSON files as fallback
            for i, chunk in enumerate(chunks):
                chunk_file = os.path.join(embedding_path, f"chunk_{i}.json")
                try:
                    with open(chunk_file, 'w', encoding='utf-8') as f:
                        json_data = {
                            'page_content': chunk.page_content,
                            'metadata': chunk.metadata,
                            'source': chunk.metadata.get('source', original_name)
                        }
                        json.dump(json_data, f, ensure_ascii=False, indent=2)
                except Exception as json_error:
                    print(f"Error saving chunk to JSON: {json_error}")
        
        # Save document info to database
        conn = sqlite3.connect('bots.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO bot_documents 
            (bot_id, filename, original_name, file_type, embedding_path)
            VALUES (?, ?, ?, ?, ?)
        ''', (bot_id, unique_filename, original_name, file_ext, embedding_path))
        conn.commit()
        conn.close()
        print("Document information saved to database")
        
        return True, f"Document '{original_name}' added to bot's knowledge base"
        
    except Exception as e:
        print(f"Error processing document: {e}")
        traceback.print_exc()
        return False, f"Error processing document: {str(e)}"

def handle_bot_file_upload(file_path, bot_name_dropdown):
    """
    Handle file upload for bot knowledge base
    
    Args:
        file_path: File path from Gradio file upload
        bot_name_dropdown (str): Selected bot name
    
    Returns:
        str: Status message
    """
    try:
        if not file_path:
            return "No file uploaded"
        
        # Get bot ID
        conn = sqlite3.connect('bots.db')
        cursor = conn.cursor()
        cursor.execute('SELECT id FROM bots WHERE name = ?', (bot_name_dropdown,))
        result = cursor.fetchone()
        
        if not result:
            # Create new bot if it doesn't exist
            cursor.execute('INSERT INTO bots (name) VALUES (?)', (bot_name_dropdown,))
            bot_id = cursor.lastrowid
            conn.commit()
        else:
            bot_id = result[0]
        
        conn.close()
        
        # Process the document
        success, message = process_bot_document(file_path, os.path.basename(file_path), bot_id)
        
        if success:
            return message
        else:
            return f"Error: {message}"
        
    except Exception as e:
        print(f"Error handling bot file upload: {e}")
        traceback.print_exc()
        return f"Error: {str(e)}"

def get_bot_documents(bot_name):
    """
    Get list of documents in a bot's knowledge base
    
    Args:
        bot_name (str): Name of the bot
    
    Returns:
        list: List of document information
    """
    try:
        conn = sqlite3.connect('bots.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT d.original_name, d.file_type, d.uploaded_at
            FROM bot_documents d
            JOIN bots b ON d.bot_id = b.id
            WHERE b.name = ?
            ORDER BY d.uploaded_at DESC
        ''', (bot_name,))
        
        documents = cursor.fetchall()
        conn.close()
        
        if documents:
            return pd.DataFrame(documents, columns=['Name', 'Type', 'Uploaded At'])
        return pd.DataFrame(columns=['Name', 'Type', 'Uploaded At'])
        
    except Exception as e:
        print(f"Error getting bot documents: {e}")
        return pd.DataFrame(columns=['Name', 'Type', 'Uploaded At'])

def get_bot_knowledge_context(bot_name, query):
    """
    Get relevant context from bot's knowledge base
    
    Args:
        bot_name (str): Name of the bot
        query (str): Query to find relevant context
    
    Returns:
        str: Relevant context from documents
    """
    try:
        if not bot_name or bot_name == 'New Bot':
            return ""
        
        # Get bot's documents
        conn = sqlite3.connect('bots.db')
        cursor = conn.cursor()
        cursor.execute('''
            SELECT d.embedding_path
            FROM bot_documents d
            JOIN bots b ON d.bot_id = b.id
            WHERE b.name = ?
        ''', (bot_name,))
        
        embedding_paths = cursor.fetchall()
        conn.close()
        
        if not embedding_paths:
            return ""
        
        # Initialize TF-IDF embeddings model
        print("Initializing TF-IDF embeddings model for bot knowledge...")
        embedding_function = TfidfEmbeddings()
        
        # Search for relevant chunks across all documents
        all_contexts = []
        for path in embedding_paths:
            try:
                # Use Chroma with TF-IDF embeddings
                db = Chroma(
                    persist_directory=path[0],
                    embedding_function=embedding_function
                )
                results = db.similarity_search(query, k=2)
                for doc in results:
                    all_contexts.append(doc.page_content)
                print(f"Found {len(results)} relevant chunks using TF-IDF search")
            except Exception as e:
                print(f"Error using TF-IDF search for bot knowledge: {e}")
                
                # Fallback to keyword-based search if TF-IDF fails
                print("Falling back to keyword-based search for bot knowledge")
                
                try:
                    # Look for document chunks in the collection directory
                    collection_dir = os.path.join(path[0], "chroma-embeddings.parquet")
                    if os.path.exists(collection_dir):
                        import pandas as pd
                        # Try to load the parquet file with document contents
                        df = pd.read_parquet(collection_dir)
                        if 'document' in df.columns:
                            # Simple keyword matching
                            query_terms = query.lower().split()
                            matched_chunks = []
                            
                            for _, row in df.iterrows():
                                content = row['document'].lower()
                                # Calculate a simple relevance score based on term frequency
                                score = sum(1 for term in query_terms if term in content)
                                if score > 0:
                                    matched_chunks.append((row['document'], score))
                            
                            # Sort by score and take top 2
                            matched_chunks.sort(key=lambda x: x[1], reverse=True)
                            for chunk, _ in matched_chunks[:2]:
                                all_contexts.append(chunk)
                                
                            print(f"Found {len(matched_chunks)} chunks using keyword search")
                    else:
                        # Try to find JSON files in the embedding directory
                        json_files = glob.glob(os.path.join(path[0], "chunk_*.json"))
                        if json_files:
                            print(f"Found {len(json_files)} JSON chunk files")
                            
                            # Load all chunks from JSON files
                            chunks = []
                            for json_file in json_files:
                                try:
                                    with open(json_file, 'r', encoding='utf-8') as f:
                                        chunk_data = json.load(f)
                                        chunks.append(chunk_data)
                                except Exception as json_error:
                                    print(f"Error loading chunk from {json_file}: {json_error}")
                            
                            # Simple keyword matching
                            query_terms = query.lower().split()
                            matched_chunks = []
                            for chunk in chunks:
                                content = chunk.get('page_content', '').lower()
                                # Calculate a simple relevance score based on term frequency
                                score = sum(1 for term in query_terms if term in content)
                                if score > 0:
                                    matched_chunks.append((chunk.get('page_content', ''), score))
                            
                            # Sort by score and take top 2
                            matched_chunks.sort(key=lambda x: x[1], reverse=True)
                            for chunk, _ in matched_chunks[:2]:
                                all_contexts.append(chunk)
                                
                            print(f"Found {len(matched_chunks)} chunks using keyword search")
                except Exception as fallback_error:
                    print(f"Error in fallback search for bot knowledge: {fallback_error}")
                continue
        
        return "\n\n".join(all_contexts) if all_contexts else ""
        
    except Exception as e:
        print(f"Error getting bot knowledge context: {e}")
        return ""

def load_existing_bots():
    """
    Load existing bots from the database into a DataFrame
    
    Args:
        None
    
    Returns:
        pd.DataFrame: DataFrame containing bot configurations
    """
    try:
        conn = sqlite3.connect('bots.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT name, base_model, system_prompt, temperature, max_tokens, top_p
            FROM bots
            ORDER BY name
        ''')
        
        bots = cursor.fetchall()
        conn.close()
        
        if bots:
            return pd.DataFrame(
                bots,
                columns=['Name', 'Base Model', 'System Prompt', 'Temperature', 'Max Tokens', 'Top P']
            )
        return pd.DataFrame(
            columns=['Name', 'Base Model', 'System Prompt', 'Temperature', 'Max Tokens', 'Top P']
        )
        
    except Exception as e:
        print(f"Error loading existing bots: {e}")
        return pd.DataFrame(
            columns=['Name', 'Base Model', 'System Prompt', 'Temperature', 'Max Tokens', 'Top P']
        )

def handle_bot_selection(bot_name):
    """
    Handle bot selection from dropdown
    
    Args:
        bot_name (str): Selected bot name
    
    Returns:
        tuple: Bot configuration parameters
    """
    try:
        if bot_name == 'New Bot':
            return (
                gr.update(visible=True),  # bot_name_input
                gr.update(value=None),    # base_model
                "",                       # system_prompt
                0.7,                      # temperature
                4096,                     # max_tokens
                0.9                       # top_p
            )
        
        # Get bot configuration
        conn = sqlite3.connect('bots.db')
        cursor = conn.cursor()
        cursor.execute('''
            SELECT base_model, system_prompt, temperature, max_tokens, top_p
            FROM bots
            WHERE name = ?
        ''', (bot_name,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return (
                gr.update(visible=False),  # bot_name_input
                gr.update(value=result[0]), # base_model
                result[1],                  # system_prompt
                result[2],                  # temperature
                result[3],                  # max_tokens
                result[4]                   # top_p
            )
        
        return (
            gr.update(visible=True),  # bot_name_input
            gr.update(value=None),    # base_model
            "",                       # system_prompt
            0.7,                      # temperature
            4096,                     # max_tokens
            0.9                       # top_p
        )
        
    except Exception as e:
        print(f"Error handling bot selection: {e}")
        return (
            gr.update(visible=True),  # bot_name_input
            gr.update(value=None),    # base_model
            "",                       # system_prompt
            0.7,                      # temperature
            4096,                     # max_tokens
            0.9                       # top_p
        )

# Initialize global comfyui variable
comfyui = None
MAX_INIT_RETRIES = 3
RETRY_DELAY = 2  # seconds

def initialize_comfyui():
    """Initialize ComfyUI and set up the global instance"""
    try:
        from comfyui_integration import ComfyUIIntegrator
        global comfyui
        
        retries = 0
        while retries < MAX_INIT_RETRIES:
            try:
                print(f"Starting ComfyUI initialization (attempt {retries + 1}/{MAX_INIT_RETRIES})...")
                comfyui = ComfyUIIntegrator()
                print("ComfyUI initialized successfully")
                return comfyui
            except Exception as e:
                retries += 1
                if retries < MAX_INIT_RETRIES:
                    print(f"Initialization attempt {retries} failed: {str(e)}")
                    print(f"Retrying in {RETRY_DELAY} seconds...")
                    import time
                    time.sleep(RETRY_DELAY)
                else:
                    raise
        
    except ImportError as e:
        print(f"Failed to import ComfyUI modules: {str(e)}")
        print("Make sure all required packages are installed")
        return None
    except Exception as e:
        print(f"Error initializing ComfyUI: {str(e)}")
        import traceback
        print("Full traceback:")
        print(traceback.format_exc())
        return None

# Start ComfyUI in a separate thread
print("Starting ComfyUI initialization thread...")
comfyui_thread = threading.Thread(target=initialize_comfyui)
comfyui_thread.daemon = True
comfyui_thread.start()

def generate_image_from_prompt(prompt, width, height, model_name=None):
    """Generate an image using ComfyUI based on the given prompt and selected workflow"""
    try:
        global comfyui
        if comfyui is None:
            print("ComfyUI is not initialized, checking initialization status...")
            # Wait briefly to see if initialization completes
            import time
            for _ in range(10):  # Try for up to 20 seconds
                time.sleep(2)
                if comfyui is not None:
                    break
            if comfyui is None:
                raise gr.Error("ComfyUI is not initialized yet. Please wait a moment and try again.")
        
        print(f"Generating image with prompt: {prompt}, size: {width}x{height}, model: {model_name}")
        
        # Convert width and height to integers
        width = int(width)
        height = int(height)
        
        # Extract workflow name if it's a workflow
        workflow_name = None
        if model_name and model_name.startswith("Workflow: "):
            workflow_name = model_name[len("Workflow: "):]
        
        # Generate the image with a longer timeout
        image = comfyui.generate_image(
            prompt, 
            width=width, 
            height=height, 
            timeout=600,  # 10-minute timeout
            workflow_name=workflow_name
        )
        
        if image is None:
            raise gr.Error("Failed to generate image. The process may have timed out or encountered an error.")
        
        print("Image generated successfully")
        return image
        
    except gr.Error as e:
        raise e
    except Exception as e:
        print(f"Error in generate_image_from_prompt: {str(e)}")
        import traceback
        print("Full traceback:")
        print(traceback.format_exc())
        raise gr.Error(f"Image generation failed: {str(e)}")

def get_available_workflows():
    """
    Get list of available workflows from the workflows directory
    
    Args:
        None
    
    Returns:
        list: List of workflow names without file extensions
    """
    try:
        global comfyui
        if comfyui is None:
            # Initialize ComfyUI if not already initialized
            comfyui = initialize_comfyui()
        
        if comfyui and hasattr(comfyui, 'api_client'):
            return comfyui.api_client.get_available_workflows()
        return []
    except Exception as e:
        print(f"Error getting workflows: {e}")
        return []

def get_all_models_and_workflows():
    """
    Get combined list of models, bots, and workflows for the model dropdown
    
    Args:
        None
    
    Returns:
        list: Combined list of models, bots, and workflows
    """
    # Get available Ollama models
    models = get_available_models()
    
    # Add SDXL as a special option
    if "SDXL" not in models:
        models.append("SDXL")
    
    # Get bot names
    bot_names = get_bot_names()
    
    # Get available workflows
    workflows = get_available_workflows()
    
    # Create a formatted list of workflow options
    workflow_options = [f"Workflow: {workflow}" for workflow in workflows]
    
    # Combine models, bots, and workflows
    return models + bot_names + workflow_options

@app.route('/generate-image', methods=['POST'])
def generate_image():
    data = request.get_json()
    prompt = data.get('prompt')
    width = data.get('width', 1024)
    height = data.get('height', 1024)
    
    if not prompt:
        return jsonify({'error': 'Prompt is required'}), 400
    
    try:
        # Generate the image
        image = comfyui.generate_image(prompt, width=width, height=height, timeout=600)  # 10-minute timeout
        image_path = f'static/generated/{str(uuid.uuid4())}.png'
        image.save(image_path)
        
        # Create or update conversation with image generation title
        title = generate_conversation_title(prompt, "", "SDXL")
        
        # Create a new conversation for the image generation
        conv_id = create_conversation(prompt, "SDXL")
        
        # Update the conversation title
        update_conversation_title(conv_id, title)
        
        return jsonify({
            'image_url': image_path, 
            'conversation_id': conv_id, 
            'conversation_title': title
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

with gr.Blocks(theme=gr.themes.Soft(), css="""
    /* Modal Styling */
    .gradio-modal {
        max-width: 800px !important;
        width: 90% !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
    }
    .gradio-modal .tabs {
        margin-top: 1em !important;
    }
    .gradio-modal .tab-nav {
        display: flex !important;
        justify-content: center !important;
        margin-bottom: 1em !important;
    }
    .gradio-modal .tab-nav button {
        margin: 0 0.5em !important;
        padding: 0.5em 1em !important;
        border-radius: 6px !important;
    }
    .gradio-modal .tab-nav button.selected {
        background-color: #4a90e2 !important;
        color: white !important;
    }
    .gradio-modal .form-row {
        display: flex !important;
        gap: 1em !important;
        margin-bottom: 1em !important;
    }
    .gradio-modal .form-column {
        flex: 1 !important;
    }
    .gradio-modal .slider-container {
        margin-bottom: 1em !important;
    }
    .gradio-modal .download-section {
        display: flex !important;
        align-items: flex-end !important;
        gap: 1em !important;
    }
    .gradio-modal .download-section .gradio-textbox {
        flex-grow: 1 !important;
    }
    .gradio-modal .default-model-section {
        margin-top: 1em !important;
        display: flex !important;
        align-items: center !important;
        gap: 1em !important;
    }
    #conv_list {
        border: none !important;
        overflow-x: hidden !important;
        margin-top: 0.25em !important;
    }
    #conv_list table {
        border: none !important;
        width: 100% !important;
        table-layout: fixed !important;
    }
    #conv_list thead {
        display: none !important;
    }
    #conv_list td {
        border: none !important;
        padding: 8px 12px !important;
        cursor: pointer !important;
        transition: background-color 0.3s !important;
        white-space: normal !important;
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
    }
    #conv_list tr:hover td {
        background-color: rgba(0, 0, 0, 0.05) !important;
    }
    #conv_list tr.selected td {
        background-color: rgba(0, 0, 0, 0.1) !important;
    }
    /* Hide scrollbar but keep functionality */
    #conv_list > div {
        scrollbar-width: none !important;
        -ms-overflow-style: none !important;
        height: calc(100vh - 370px) !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
    }
    #conv_list > div::-webkit-scrollbar {
        display: none !important;
    }
    .gradio-container {
        max-width: 100% !important;
    }
    .gradio-footer, footer {
        display: none !important;
    }
    #xeno_logo {
        margin: 0.25em auto !important;
        display: block !important;
        max-width: 60% !important;
        padding: 0 !important;
    }
    #xeno_logo > div {
        border: none !important;
        background: none !important;
        box-shadow: none !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    #xeno_logo img {
        object-fit: contain !important;
        margin: 0 auto !important;
        display: block !important;
        width: 250px !important;
        height: auto !important;
        max-height: 150px !important;
    }
    /* Adjust spacing for sidebar elements */
    #new_chat_btn {
        margin: 0.25em 0 !important;
    }
    #search_bar {
        margin: 0.25em 0 !important;
    }
    #manage_bots_btn {
        margin: 4px 0 0 0 !important;  /* Increased spacing */
        width: 100% !important;
    }
    #app_footer {
        text-align: center !important;
        width: 100% !important;
        padding: 0.5em 0 !important;
        margin-top: 0.5em !important;
        color: #666 !important;
        font-size: 0.8em !important;
        border-top: 1px solid #eee !important;
    }
    /* Dialog Styling */
    .gradio-container .dialog-overlay {
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        width: 100% !important;
        height: 100% !important;
        background-color: rgba(0, 0, 0, 0.5) !important;
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        z-index: 1000 !important;
    }
    .gradio-container .dialog-content {
        background-color: white !important;
        max-width: 800px !important;
        width: 90% !important;
        max-height: 80vh !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2) !important;
        padding: 1.5em !important;
        overflow-y: auto !important;
    }
    .gradio-container .dialog-tabs {
        margin-bottom: 1em !important;
    }
    .gradio-container .dialog-tabs .tab-nav {
        display: flex !important;
        justify-content: center !important;
        margin-bottom: 1em !important;
    }
    .gradio-container .dialog-tabs .tab-nav button {
        margin: 0 0.5em !important;
        padding: 0.5em 1em !important;
        border-radius: 6px !important;
    }
    .gradio-container .dialog-tabs .tab-nav button.selected {
        background-color: #4a90e2 !important;
        color: white !important;
    }
    .gradio-container .dialog-form-row {
        display: flex !important;
        gap: 1em !important;
        margin-bottom: 1em !important;
    }
    .gradio-container .dialog-form-column {
        flex: 1 !important;
    }
    .gradio-container .dialog-slider-container {
        margin-bottom: 1em !important;
    }
    .gradio-container .dialog-download-section {
        display: flex !important;
        align-items: flex-end !important;
        gap: 1em !important;
    }
    .gradio-container .dialog-download-section .gradio-textbox {
        flex-grow: 1 !important;
    }
    .gradio-container .dialog-default-model-section {
        margin-top: 1em !important;
        display: flex !important;
        align-items: center !important;
        gap: 1em !important;
    }
    /* Button hover animations */
    .round-button {
        border-radius: 50% !important;
        padding: 8px !important;
        min-width: 40px !important;
        max-width: 40px !important;
        height: 40px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        background-color: transparent !important;
        border: 1px solid #ccc !important;
        transition: all 0.3s ease !important;
        cursor: pointer !important;
    }
    
    .round-button:hover {
        transform: scale(1.1) !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
        background-color: rgba(0, 0, 0, 0.05) !important;
    }
    
    .round-button:active {
        transform: scale(0.95) !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
    }
""", title="MIDAS 2.0") as demo:

    current_conversation = gr.State()
    current_model = gr.State()
    overlay_visible = gr.State(False)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Image("xeno.png", 
            label=None, 
            show_label=False, 
            container=False, 
            height=150, 
            elem_classes=["xeno-logo-container", "xeno-logo"], 
            show_download_button=False, 
            show_fullscreen_button=False)
            # Manage Bots button
            gr.HTML(f"""
                <style>
                    /* Manage Bots Button Styling */
                    #manage_bots_btn {{
                        display: flex;
                        align-items: center;
                        gap: 10px;
                        padding: 8px 12px !important;
                    }}
                    #manage_bots_btn img {{
                        width: 24px;
                        height: 24px;
                        object-fit: contain;
                    }}
                </style>
            """)
            
            manage_bots_btn = gr.Button(
                value="Manage Bots", 
                elem_id="manage_bots_btn", 
                icon=os.path.join(os.path.dirname(__file__), "ui-assets", "bot-setting.png")
            )
            
            # New chat button
            gr.HTML(f"""
                <style>
                    /* New Chat Button Styling */
                    #new_chat_btn {{
                        display: flex;
                        align-items: center;
                        gap: 10px;
                        padding: 8px 12px !important;
                        margin-top: 10px !important;  /* Increased spacing */
                    }}
                    #new_chat_btn img {{
                        width: 24px;
                        height: 24px;
                        object-fit: contain;
                    }}
                </style>
            """)
            
            new_conv_btn = gr.Button(
                value="New Chat", 
                elem_id="new_chat_btn", 
                icon=os.path.join(os.path.dirname(__file__), "ui-assets", "chat.png")
            )
            
            # Search bar for conversations
            search_bar = gr.Textbox(
                placeholder="Search conversations...", 
                show_label=False,
                container=False
            )
            
            # Conversation list
            conv_list = gr.Dataframe(
                headers=["title"],
                datatype=["str"],
                row_count=5,
                interactive=False,
                wrap=False,
                column_widths=["100%"],
                elem_id="conv_list"
            )
        
        with gr.Column(scale=3):
            # Get available models and format them
            available_models = [get_model_display_name(m) for m in get_available_models()]  # Ollama models
            bot_names = get_bot_names()  # Bot names
            all_models = available_models + bot_names + ["SDXL"]  # Combine all models
            
            model_dropdown = gr.Dropdown(
                choices=get_all_models_and_workflows(),
                value=get_default_model(),  # Use the default model from settings
                label="Model",
                show_label=False,
                interactive=True,
                elem_id="model_selector"
            )
            
            # Chat title at the top
            with gr.Row():
                chat_title = gr.Textbox(
                    label="Conversation", 
                    interactive=False,  # Make non-editable by default
                    show_label=False,
                    container=False,
                    elem_id="chat_title",
                    scale=5
                )
                rename_conv_btn = gr.Button("", variant="secondary", scale=0.5, icon=os.path.join(os.path.dirname(__file__), "ui-assets", "edit.png"))
                delete_conv_btn = gr.Button("", variant="stop", scale=0.5, icon=os.path.join(os.path.dirname(__file__), "ui-assets", "delete.png"))
                confirm_rename_btn = gr.Button("Save", variant="primary", scale=0.5, visible=False)  # Save button
                
            # Rename button click logic
            def toggle_rename_mode(current_title):
                return {
                    rename_conv_btn: gr.update(visible=False),
                    confirm_rename_btn: gr.update(visible=True),
                    chat_title: gr.update(interactive=True)
                }
            
            def confirm_rename(current_conversation, new_title):
                if not new_title:
                    return {
                        chat_title: gr.update(interactive=False),
                        rename_conv_btn: gr.update(visible=True),
                        confirm_rename_btn: gr.update(visible=False),
                        conv_list: format_conversation_list()
                    }
                
                # Rename the conversation
                rename_current_chat(current_conversation, new_title)
                
                return {
                    chat_title: gr.update(interactive=False, value=new_title),
                    rename_conv_btn: gr.update(visible=True),
                    confirm_rename_btn: gr.update(visible=False),
                    conv_list: format_conversation_list()
                }
            
            # Bind rename functionality
            rename_conv_btn.click(
                fn=toggle_rename_mode,
                inputs=[chat_title],
                outputs=[rename_conv_btn, confirm_rename_btn, chat_title]
            )
            
            confirm_rename_btn.click(
                fn=confirm_rename,
                inputs=[current_conversation, chat_title],
                outputs=[chat_title, rename_conv_btn, confirm_rename_btn, conv_list]
            )
            
            # Conversation options right under the title
            with gr.Row():
                pass

            # Chatbot interface
            chatbot = gr.Chatbot(
                label="Conversation", 
                show_label=False, 
                height=600,
                layout="bubble",
                bubble_full_width=False,
                value=[("Welcome to MIDAS 2.0, press New Chat and select a model to start chatting!", None)]
            )
            
            # Message input area with integrated controls
            with gr.Row(variant="panel"):
                with gr.Column(scale=12):
                    # Document info overlay (hidden by default)
                    with gr.Row(visible=False, elem_classes=["doc-info-overlay"]) as doc_info_overlay:
                        with gr.Column():
                            gr.Markdown("📄 **Document uploaded**")
                            with gr.Row():
                                doc_name_display = gr.Markdown("")
                                remove_btn = gr.Button("❌", elem_classes=["remove-button"])

                    # Chat input
                    msg_box = gr.Textbox(
                        show_label=False,
                        placeholder="Enter text and press enter, or upload a document",
                        elem_id="input_box"
                    )
                    
                    # Chat control buttons centered under the input bar
                    with gr.Row(elem_classes=["round-button-row"]):
                        with gr.Column(scale=11):
                            pass  # Placeholder to push buttons to the right
                        with gr.Column(scale=1):
                            with gr.Row(elem_classes=["buttons-row"]):
                                file_upload = gr.UploadButton(
                                    "",  # Empty text since we'll use image
                                    file_types=[".pdf", ".txt", ".doc", ".docx"],
                                    label="Add Attachment",
                                    size="sm",
                                    elem_id="file_upload_button",
                                    elem_classes=["round-button", "image-button", "tooltip"]
                                )
                                submit_btn = gr.Button(
                                    "",  # Empty text since we'll use image
                                    variant="primary",
                                    size="sm",
                                    elem_id="send_button",
                                    elem_classes=["round-button", "image-button", "tooltip"]
                                )
                                stop_btn = gr.Button(
                                    "",  # Empty text since we'll use image
                                    variant="stop",
                                    size="sm",
                                    elem_id="stop_button",
                                    elem_classes=["round-button", "image-button", "tooltip"],
                                    visible=False
                                )
                                undo_btn = gr.Button(
                                    "",  # Empty text since we'll use image
                                    variant="secondary",
                                    size="sm",
                                    elem_id="undo_button",
                                    elem_classes=["round-button", "image-button", "tooltip"]
                                )
                    
                    gr.HTML("""
                        <style>
                            .tooltip::before {
                                content: attr(data-tooltip);
                                position: absolute;
                                bottom: 100%;
                                left: 50%;
                                transform: translateX(-50%);
                                padding: 5px;
                                background-color: #333;
                                color: white;
                                border-radius: 4px;
                                font-size: 12px;
                                white-space: nowrap;
                                opacity: 0;
                                transition: opacity 0.3s;
                            }
                            .tooltip:hover::before {
                                opacity: 1;
                            }
                            #file_upload_button::before { content: "Add Attachment"; }
                            #send_button::before { content: "Send Message"; }
                            #stop_button::before { content: "Stop Generation"; }
                            #undo_button::before { content: "Undo Last Message"; }
                        </style>
                    """)
            
            # Upload status (hidden by default)
            upload_status = gr.Markdown(visible=False)
            
            # Add custom CSS for round buttons, layout and overlay
            file_upload_base64 = base64.b64encode(open("ui-assets/file-upload.png", "rb").read()).decode('utf-8')
            send_base64 = base64.b64encode(open("ui-assets/send.png", "rb").read()).decode('utf-8')
            stop_base64 = base64.b64encode(open("ui-assets/stop.png", "rb").read()).decode('utf-8')
            undo_base64 = base64.b64encode(open("ui-assets/undo.png", "rb").read()).decode('utf-8')
            gr.HTML(f"""
                <style>
                    /* Round buttons */
                    .round-button {{
                        border-radius: 50% !important;
                        padding: 8px !important;
                        min-width: 40px !important;
                        max-width: 40px !important;
                        height: 40px !important;
                        display: flex !important;
                        align-items: center !important;
                        justify-content: center !important;
                        background-color: transparent !important;
                        border: 1px solid #ccc !important;
                        transition: all 0.3s ease !important;
                        cursor: pointer !important;
                    }}
                    
                    /* Image buttons */
                    .image-button {{
                        background-size: 20px 20px !important;
                        background-repeat: no-repeat !important;
                        background-position: center !important;
                    }}
                    
                    #file_upload_button {{
                        background-image: url('data:image/png;base64,{file_upload_base64}') !important;
                    }}
                    
                    #send_button {{
                        background-image: url('data:image/png;base64,{send_base64}') !important;
                    }}
                    
                    #stop_button {{
                        background-image: url('data:image/png;base64,{stop_base64}') !important;
                    }}
                    
                    #undo_button {{
                        background-image: url('data:image/png;base64,{undo_base64}') !important;
                    }}
                    
                    /* Button row styling */
                    .buttons-row {{
                        display: flex !important;
                        justify-content: flex-end !important;
                        align-items: center !important;
                        gap: 10px !important;
                    }}
                </style>
            """)
    
    # Manage Bots Dialog
    manage_bots_dialog = gr.Group(visible=False)
    
    with manage_bots_dialog:
        with gr.Tabs():
            # Bot Management Tab
            with gr.Tab("Bot Management"):
                with gr.Row():
                    bot_name_dropdown = gr.Dropdown(
                        choices=get_existing_bots() + ['New Bot'], 
                        label="Select or Create Bot",
                        value='New Bot'
                    )
                    bot_name_input = gr.Textbox(
                        label="Custom Bot Name", 
                        placeholder="Enter custom bot name...", 
                        visible=True
                    )
                    base_model = gr.Dropdown(
                        choices=[get_model_display_name(m) for m in get_available_models()],
                        label="Base Model"
                    )
                
                system_prompt = gr.Textbox(
                    label="System Prompt",
                    lines=3,
                    placeholder="Enter system prompt..."
                )
                
                with gr.Row():
                    temperature = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=0.7,
                        label="Temperature"
                    )
                    max_tokens = gr.Number(
                        value=4096,
                        label="Max Tokens"
                    )
                    top_p = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.9,
                        label="Top P"
                    )
                
                with gr.Row():
                    save_status = gr.Textbox(
                        label="Status",
                        interactive=False
                    )
                
                save_bot_btn = gr.Button("Save Bot Configuration")
                delete_bot_btn = gr.Button("Delete Bot Configuration")
                
                # Existing Bots Display
                existing_bots_df = gr.DataFrame(
                    value=load_existing_bots(),
                    label="Existing Bots",
                    interactive=False,
                    wrap=True
                )
                
                # Knowledge Base Section
                gr.Markdown("### Knowledge Base")
                with gr.Row():
                    knowledge_file_upload = gr.File(
                        label="Upload Document",
                        file_types=[".txt", ".pdf", ".docx"],
                        type="filepath"
                    )
                    knowledge_upload_status = gr.Textbox(label="Upload Status", interactive=False)
                
                # Document List
                documents_df = gr.DataFrame(
                    value=pd.DataFrame(columns=['Name', 'Type', 'Uploaded At']),
                    label="Bot Documents",
                    interactive=False,
                    wrap=True
                )
                
                # Event Handlers
                bot_name_dropdown.change(
                    fn=handle_bot_selection,
                    inputs=[bot_name_dropdown],
                    outputs=[
                        bot_name_input,
                        base_model,
                        system_prompt,
                        temperature,
                        max_tokens,
                        top_p
                    ]
                ).then(  # Chain with document list update
                    fn=lambda bot_name: get_bot_documents(bot_name),
                    inputs=[bot_name_dropdown],
                    outputs=[documents_df]
                )
                
                save_bot_btn.click(
                    fn=handle_bot_save,
                    inputs=[
                        bot_name_dropdown,
                        bot_name_input,
                        base_model,
                        system_prompt,
                        temperature,
                        max_tokens,
                        top_p,
                        model_dropdown
                    ],
                    outputs=[
                        existing_bots_df,  # Existing bots list
                        bot_name_dropdown,  # Bot name dropdown update
                        base_model,        # Base model dropdown update
                        save_status,       # Save status
                        model_dropdown     # Model dropdown update
                    ]
                )
                
                delete_bot_btn.click(
                    fn=delete_bot_configuration,
                    inputs=[bot_name_dropdown],
                    outputs=[
                        existing_bots_df,  # Existing bots list
                        bot_name_dropdown,  # Bot name dropdown
                        save_status        # Save status
                    ]
                )
                
                # File upload handler
                knowledge_file_upload.upload(
                    fn=handle_bot_file_upload,
                    inputs=[
                        knowledge_file_upload,
                        bot_name_dropdown
                    ],
                    outputs=[knowledge_upload_status]
                ).then(  # Chain with document list update
                    fn=lambda bot_name: get_bot_documents(bot_name),
                    inputs=[bot_name_dropdown],
                    outputs=[documents_df]
                )
            
            # Models Tab
            with gr.Tab("Models"):
                ollama_models = gr.DataFrame(
                    load_ollama_models(),
                    label="Available Models",
                    interactive=False,
                    wrap=True
                )
                download_model_input = gr.Textbox(label="Model Name", placeholder="Enter model name (e.g. llama3.1:8b)")
                download_model = gr.Button("Download Model")
                download_status = gr.Textbox(label="Download Status", interactive=False)
        
        # Connect the save button
        save_bot_btn.click(
            fn=handle_bot_save,
            inputs=[
                bot_name_dropdown,
                bot_name_input,
                base_model,
                system_prompt,
                temperature,
                max_tokens,
                top_p,
                model_dropdown
            ],
            outputs=[
                existing_bots_df,  # Existing bots list
                bot_name_dropdown,  # Bot name dropdown
                base_model,        # Base model dropdown
                save_status,       # Save status
                model_dropdown     # Model dropdown update
            ]
        )
        
        download_model.click(
            fn=download_ollama_model,
            inputs=[download_model_input],
            outputs=[download_status]
        )
    
    # Wire up the Manage Bots button to show dialog and load models
    manage_bots_btn.click(
        fn=open_manage_bots_dialog,
        inputs=None, 
        outputs=[
            manage_bots_dialog,  # Dialog visibility
            existing_bots_df,    # Existing bots list
            bot_name_dropdown,  # Bot name dropdown
            base_model,          # Base model dropdown
            save_status          # Save status
        ]
    )

    # Connect the close button
    close_dialog_btn = gr.Button("Close", variant="secondary")
    close_dialog_btn.click(
        fn=lambda: (gr.update(visible=False), gr.update(visible=False)),
        outputs=[manage_bots_dialog, close_dialog_btn]
    )

    # Event handlers
    demo.load(
        format_conversation_list, 
        outputs=[conv_list]
    )
    
    new_conv_btn.click(
        lambda: [
            None,  # current_conversation
            [],    # chatbot
            get_default_model(),  # model
            "Untitled Conversation",  # initial title
            format_conversation_list()  # conversation list
        ],
        outputs=[
            current_conversation, 
            chatbot, 
            model_dropdown, 
            chat_title, 
            conv_list
        ]
    )
    
    
    def submit_message(message, history, conv_id, model_name):
        """
        Submit a message to the chat, with special handling for SDXL image generation
        
        Args:
            message (str): User's input message
            history (list): Current conversation history
            conv_id (str): Conversation ID
            model_name (str): Selected model/mode
        
        Returns:
        tuple: Updated outputs for Gradio interface
        """
        # Ensure conversation ID is valid
        if not conv_id:
            conv_id = create_conversation(message, model_name)
        
        # Get relevant context from documents
        context = get_relevant_context(message, conv_id)
        print(f"Retrieved context for conversation {conv_id}")
        
        def parse_sdxl_commands(message):
            """Parse SDXL commands from the message and return parameters and cleaned message."""
            params = {
                'width': 1024,
                'height': 1024,
                'negative_prompt': "",
                'style': None,
                'quality': 1.0,
                'steps': 30,
                'cfg': 7.0,
                'seed': None  # None means use random seed
            }
            
            # Define aspect ratio presets
            aspect_ratios = {
                'square': (1024, 1024),
                'portrait': (832, 1216),
                'landscape': (1216, 832),
                'wide': (1344, 768),
                'tall': (768, 1344)
            }
            
            # Define style presets
            style_presets = {
                'photo': "detailed photograph, 4k, high quality, photorealistic",
                'anime': "anime style, high quality, detailed, vibrant colors",
                'painting': "digital painting, detailed brushwork, artistic, high quality",
                'sketch': "pencil sketch, detailed linework, artistic",
                '3d': "3D render, octane render, high quality, detailed lighting",
                'cinematic': "cinematic shot, dramatic lighting, movie quality, 4k"
            }
            
            # Split message into prompt and commands
            parts = message.split('--')
            clean_prompt = parts[0].strip()
            
            # Process each command
            for part in parts[1:]:
                if not part.strip():
                    continue
                
                # Split command and value
                cmd_parts = part.strip().split(None, 1)
                if not cmd_parts:
                    continue
                
                cmd = cmd_parts[0].lower()
                value = cmd_parts[1] if len(cmd_parts) > 1 else ''
                
                if cmd == 'ar' and value in aspect_ratios:
                    params['width'], params['height'] = aspect_ratios[value]
                elif cmd == 'style' and value in style_presets:
                    params['style'] = style_presets[value]
                elif cmd == 'quality' and value:
                    try:
                        params['quality'] = float(value)
                        params['quality'] = max(0.1, min(2.0, params['quality']))
                    except ValueError:
                        pass
                elif cmd == 'steps' and value:
                    try:
                        params['steps'] = int(value)
                        params['steps'] = max(1, min(100, params['steps']))
                    except ValueError:
                        pass
                elif cmd == 'cfg' and value:
                    try:
                        params['cfg'] = float(value)
                        params['cfg'] = max(1.0, min(20.0, params['cfg']))
                    except ValueError:
                        pass
                elif cmd == 'seed' and value:
                    try:
                        params['seed'] = int(value)
                    except ValueError:
                        pass
                elif cmd == 'neg':
                    params['negative_prompt'] = value
    
            return params, clean_prompt

        # Handle SDXL image generation
        if model_name == "SDXL" or model_name.startswith("Workflow: "):
            try:
                # Parse SDXL commands
                sdxl_params, clean_prompt = parse_sdxl_commands(message)
                
                # Generate image using ComfyUI
                global comfyui
                if comfyui is None:
                    comfyui = initialize_comfyui()
                
                # Apply style if specified
                if sdxl_params.get('style'):
                    clean_prompt = f"{clean_prompt} {sdxl_params['style']}"
                
                # Add user message to history first
                history.append((message, "Generating image..."))
                yield history, "", get_chat_title(conv_id), format_conversation_list()
                
                image = comfyui.generate_image(
                    workflow_name=model_name[len("Workflow: "):] if model_name.startswith("Workflow: ") else None,
                    prompt=clean_prompt,
                    width=sdxl_params.get('width', 1024),
                    height=sdxl_params.get('height', 1024),
                    negative_prompt=sdxl_params.get('negative_prompt', ""),
                    steps=sdxl_params.get('steps', 15),
                    cfg=sdxl_params.get('cfg', 7.0),
                    quality=sdxl_params.get('quality', 1.0),
                    seed=sdxl_params.get('seed')
                )
                
                # Ensure conversations directory exists
                os.makedirs('conversations', exist_ok=True)
                
                # Define image filename using datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_filename = f'conversations/{conv_id}/generated_image_{timestamp}.png'
                
                # Ensure the directory exists
                os.makedirs(os.path.dirname(image_filename), exist_ok=True)
                
                # Save the image
                image.save(image_filename)
                
                # Check if this is a new conversation with default title
                conn = sqlite3.connect('conversations.db')
                c = conn.cursor()
                c.execute('SELECT title FROM conversations WHERE id = ?', (conv_id,))
                current_title = c.fetchone()
                conn.close()
                
                if current_title and (not current_title[0] or current_title[0].strip() == "Untitled Conversation"):
                    # Generate title using the default model
                    title = generate_conversation_title(message, "", get_default_model())
                    update_conversation_title(conv_id, title)
                
                # Update history with the actual image
                history[-1] = (message, f"![Generated Image](/gradio_api/file={image_filename})")
                
                # Save messages to database
                save_message(conv_id, model_name, "user", message)
                save_message(conv_id, model_name, "assistant", history[-1][1])
                
                yield history, "", get_chat_title(conv_id), format_conversation_list()
                
            except Exception as e:
                error_msg = f"Image generation failed: {str(e)}"
                print(error_msg)
                
                # Update history with error
                if len(history) > 0 and history[-1][1] == "Generating image...":
                    history[-1] = (message, error_msg)
                else:
                    history.append((message, error_msg))
                
                # Save error messages
                save_message(conv_id, model_name, "user", message)
                save_message(conv_id, model_name, "assistant", error_msg)
                
                yield history, "", get_chat_title(conv_id), format_conversation_list()
        
        # Default behavior for non-SDXL models
        else:
            try:
                # Generate response using the model with context
                response_generator = generate_response(message, model_name, history, conv_id, context=context)
                
                # Add user message to history
                history.append((message, ""))
                save_message(conv_id, model_name, "user", message)
                
                yield history, "", get_chat_title(conv_id), format_conversation_list()
                
                # Stream the response
                full_response = ""
                for response in response_generator:
                    if response and isinstance(response, tuple):
                        _, updated_history, current_conv_id = response
                        # Update the response in history
                        if len(updated_history) > 0:
                            full_response = updated_history[-1][1]
                            history[-1] = (message, full_response)
                            yield history, "", get_chat_title(current_conv_id), format_conversation_list()
                    elif response and isinstance(response, str):
                        # Yield thinking process directly
                        history[-1] = (message, response)
                        yield history, "", get_chat_title(conv_id), format_conversation_list()
            
                # Save the final response
                save_message(conv_id, model_name, "assistant", full_response)
                
                # Check if this is a new conversation with default title
                conn = sqlite3.connect('conversations.db')
                c = conn.cursor()
                c.execute('SELECT title FROM conversations WHERE id = ?', (conv_id,))
                current_title = c.fetchone()
                conn.close()
                
                if current_title and (not current_title[0] or current_title[0].strip() == "Untitled Conversation"):
                    # Generate title using the default model
                    title = generate_conversation_title(message, full_response, model_name)
                    update_conversation_title(conv_id, title)
                    update_conversation_model(model_name, conv_id)
                
                yield history, "", get_chat_title(conv_id), format_conversation_list()
                
            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                print(error_msg)
                print(traceback.format_exc())
                history.append((message, error_msg))
                save_message(conv_id, model_name, "assistant", error_msg)
                yield history, "", get_chat_title(conv_id), format_conversation_list()
        
    # Add global variable to track generation state
    generation_active = False

    def stop_generation():
        global generation_active
        generation_active = False
        return "Generation stopped"

    submit_btn.click(
        fn=lambda: gr.update(visible=True),
        outputs=[stop_btn]
    )

    stop_btn.click(
        fn=stop_generation,
        outputs=[msg_box]
    )

    submit_btn.click(
        submit_message,
        inputs=[
            msg_box,           # User message input
            chatbot,           # Current chat history
            current_conversation,  # Current conversation ID
            model_dropdown     # Selected model
        ],
        outputs=[
            chatbot,           # Updated chat history
            msg_box,           # Clear message input
            chat_title,        # Update chat title
            conv_list          # Update conversation list
        ],
        queue=True
    )

    msg_box.submit(
        submit_message,
        inputs=[msg_box, chatbot, current_conversation, model_dropdown],
        outputs=[chatbot, msg_box, chat_title, conv_list],
        queue=True
    )

    delete_conv_btn.click(
        delete_conversation, 
        inputs=[current_conversation],
        outputs=[conv_list, chatbot, chat_title, current_conversation, model_dropdown]
    )

    search_bar.change(
        lambda q: format_conversation_list(q),
        inputs=[search_bar],
        outputs=[conv_list]
    )

    conv_list.select(
        handle_conversation_click, 
        inputs=[conv_list], 
        outputs=[
            current_conversation, 
            chatbot, 
            model_dropdown, 
            chat_title
        ]
    )

    model_dropdown.change(
        update_conversation_model,
        inputs=[model_dropdown, current_conversation],
        outputs=[model_dropdown]
    )

    # Add undo button click event
    def undo(chat_history, conv_id, model_name):
        if len(chat_history) >= 2:
            # Create a copy of the chat history without the last two messages
            new_history = chat_history.copy()
            del new_history[-2:]  # Remove last two messages
            save_message(conv_id, model_name, "assistant", "", full_history=new_history)
            return new_history
        return chat_history

    undo_btn.click(
        undo,
        inputs=[chatbot, current_conversation, model_dropdown],
        outputs=[chatbot]
    )

    # Add footer
    with gr.Row():
        gr.Markdown(
            "**Developed by Xenovative.Ltd, 2025**", 
            elem_id="app_footer"
        )

    # Wire up bot save functionality
    save_bot_btn.click(
        fn=handle_bot_save,
        inputs=[
            bot_name_dropdown,
            bot_name_input,
            base_model,
            system_prompt,
            temperature,
            max_tokens,
            top_p,
            model_dropdown
        ],
        outputs=[
            existing_bots_df,  # Existing bots list
            bot_name_dropdown,  # Bot name dropdown
            base_model,        # Base model dropdown
            save_status,       # Save status
            model_dropdown     # Model dropdown update
        ]
    )
    
    # Refresh existing bots list after saving
    manage_bots_btn.click(
        fn=lambda: (
            gr.update(visible=True),  # Dialog visibility
            load_existing_bots(),  # Existing bots list
            gr.update(choices=get_existing_bots() + ['New Bot'], value='New Bot'),  # Bot name dropdown
            gr.update(choices=get_available_models(), value=get_default_model()),  # Base model dropdown
            "",  # Clear save status
            gr.update(visible=True)  # Close button visibility
        ),
        inputs=None,
        outputs=[
            manage_bots_dialog,  # Dialog visibility
            existing_bots_df,    # Existing bots list
            bot_name_dropdown,   # Bot name dropdown
            base_model,          # Base model dropdown
            save_status,         # Save status
            close_dialog_btn     # Close button visibility
        ]
    )

    file_upload.upload(
        fn=handle_file_upload,
        inputs=[file_upload, current_conversation],
        outputs=[upload_status, doc_name_display, overlay_visible, current_conversation]
    ).then(
        fn=lambda x: gr.update(visible=x),
        inputs=[overlay_visible],
        outputs=[doc_info_overlay]
    )

    delete_bot_btn.click(
        fn=delete_bot_configuration,
        inputs=[bot_name_dropdown],
        outputs=[
            existing_bots_df,  # Existing bots list
            bot_name_dropdown,  # Bot name dropdown
            save_status        # Save status
        ]
    )

    bot_name_dropdown.change(
        fn=handle_bot_selection,
        inputs=[bot_name_dropdown],
        outputs=[
            bot_name_input,
            base_model,
            system_prompt,
            temperature,
            max_tokens,
            top_p
        ]
    )

def remove_document(filename, conversation_id=None):
        """Remove a document from the RAG system
    
        Args:
            filename (str): Name of the file to remove
            conversation_id (str, optional): Current conversation ID
        
        Returns:
            tuple: (status_message, document_name, visibility_state)
        """
        try:
            if not filename or not conversation_id:
                return "No document selected", "", False
                
            # Mark document as deleted in database
            conn = sqlite3.connect('conversations.db')
            c = conn.cursor()
            
            # Get document info
            c.execute('''
                SELECT id, filename, embedding_path 
                FROM documents 
                WHERE original_name = ? AND conversation_id = ?
            ''', (filename.replace("**", ""), conversation_id))
            
            doc = c.fetchone()
            if not doc:
                conn.close()
                return f"Document not found: {filename}", "", False
                
            doc_id, file_path, embedding_path = doc
            
            # Update document status
            c.execute('''
                UPDATE documents 
                SET deleted = 1 
                WHERE id = ?
            ''', (doc_id,))
            
            if c.rowcount == 0:
                conn.close()
                return f"Document not found: {filename}", "", False
            
            conn.commit()
            conn.close()
            
            return "Document marked for deletion", "", False
            
        except Exception as e:
            print(f"Error marking document for deletion: {e}")
            traceback.print_exc()
            return f"Error marking document for deletion: {str(e)}", filename, True

with gr.Blocks():
    remove_btn.click(
        fn=remove_document,
        inputs=[doc_name_display, current_conversation],
        outputs=[upload_status, doc_name_display, overlay_visible]
    ).then(
        fn=lambda x: gr.update(visible=x),
        inputs=[overlay_visible],
        outputs=[doc_info_overlay]
    )

if __name__ == "__main__":
    # Initialize databases
    init_db()
    init_bots_db()
    
    # Create required directories
    os.makedirs('bot_documents', exist_ok=True)
    os.makedirs('bot_embeddings', exist_ok=True)
    
    # Start the Gradio interface
    demo.queue().launch(server_name="0.0.0.0", share=False, allowed_paths=[".", "/conversations"], favicon_path="favicon.ico")

def cleanup_deleted_documents():
    """Clean up documents marked for deletion"""
    try:
        conn = sqlite3.connect('conversations.db')
        c = conn.cursor()
        
        # Get all deleted documents
        c.execute('''
            SELECT id, filename, embedding_path 
            FROM documents 
            WHERE deleted = 1
        ''')
        
        deleted_docs = c.fetchall()
        
        for doc_id, file_path, embedding_path in deleted_docs:
            try:
                # Remove files if they exist
                if os.path.exists(file_path):
                    os.remove(file_path)
                if os.path.exists(embedding_path):
                    shutil.rmtree(os.path.dirname(embedding_path))
                    
                # Remove from database
                c.execute('DELETE FROM documents WHERE id = ?', (doc_id,))
                
            except Exception as e:
                print(f"Warning: Error cleaning up document {doc_id}: {e}")
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        print(f"Error during document cleanup: {e}")
        traceback.print_exc()

# Register cleanup functions to run at exit
atexit.register(cleanup_deleted_documents)

def cleanup_temp_files():
    """Clean up temporary files and directories"""
    try:
        temp_dir = os.path.join(os.getcwd(), 'temp_uploads')
        if os.path.exists(temp_dir):
            print(f"Cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir)
            
        # Clear temporary documents dictionary
        temp_documents.clear()
        
    except Exception as e:
        print(f"Error cleaning up temporary files: {e}")
        traceback.print_exc()

# Register cleanup functions to run at exit
atexit.register(cleanup_deleted_documents)
atexit.register(cleanup_temp_files)

# Global dictionary to store temporary documents
temp_documents = {}

gr.HTML(f"""
    <style>
        /* Minimalistic and Round Scrollbar Styling - Global Styles */
        * {{
            scrollbar-width: thin;  /* For Firefox */
            scrollbar-color: rgba(128, 128, 128, 0.5) transparent;  /* For Firefox */
        }}

        /* Webkit (Chrome, Safari, newer versions of Opera) */
        *::-webkit-scrollbar {{
            width: 8px;  /* Thin scrollbar */
            height: 8px;  /* Horizontal scrollbar */
        }}

        *::-webkit-scrollbar-track {{
            background: transparent;  /* Transparent track */
            border-radius: 10px;
        }}

        *::-webkit-scrollbar-thumb {{
            background-color: rgba(128, 128, 128, 0.5);  /* Semi-transparent gray */
            border-radius: 10px;  /* Fully rounded scrollbar */
            border: 2px solid transparent;  /* Creates a slight padding effect */
            background-clip: content-box;  /* Ensures border doesn't affect size */
        }}

        *::-webkit-scrollbar-thumb:hover {{
            background-color: rgba(128, 128, 128, 0.7);  /* Slightly darker on hover */
        }}

        /* Ensure specific areas like conversation list and chat history have scrollbars */
        #conversations_list,
        .gradio-container .chatbot,
        .gradio-container .upload-container {{
            scrollbar-width: thin !important;
            scrollbar-color: rgba(128, 128, 128, 0.5) transparent !important;
            -ms-overflow-style: auto !important; /* Show scrollbar in Edge */
        }}
        
        /* Override any scrollbar hiding for these elements */
        #conversations_list::-webkit-scrollbar,
        .gradio-container .chatbot::-webkit-scrollbar,
        .gradio-container .upload-container::-webkit-scrollbar {{
            display: block !important;
            width: 8px !important;
        }}
        
        /* Specific exception for conversation list if needed */
        #conv_list > div {{
            scrollbar-width: thin !important;
            -ms-overflow-style: auto !important;
            height: calc(100vh - 370px) !important;
            overflow-y: auto !important;
            overflow-x: hidden !important;
        }}
        
        #conv_list > div::-webkit-scrollbar {{
            display: block !important;
            width: 8px !important;
        }}
    </style>
""")

gr.HTML(f"""
    <style>
        /* Xeno logo styling */
        .xeno-logo-container {{
            margin-bottom: 5px !important;  /* Reduced vertical spacing */
        }}
        .xeno-logo {{
            width: 250px !important;
            height: auto !important;
            max-height: 150px !important;
        }}
        
        /* New Chat Button Styling */
        #new_chat_btn {{
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 8px 12px !important;
            margin-top: 10px !important;  /* Increased spacing */
        }}
        #new_chat_btn img {{
            width: 24px;
            height: 24px;
            object-fit: contain;
        }}
    </style>
""")

gr.HTML(f"""
    <style>
        /* Conversation List and Manage Bots Button Spacing */
        .conversations-container {{
            gap: 2px !important;  /* Reduced gap between elements */
        }}
        #conversations_list {{
            margin-bottom: 0 !important;  /* Remove bottom margin */
        }}
        #manage_bots_btn {{
            margin-top: 4px !important;  /* Increased spacing */
            padding: 6px 10px !important;  /* Reduce button padding */
        }}
    </style>
""")

gr.HTML(f"""
    <style>
        /* Manage Bots Button Styling */
        #manage_bots_btn {{
            margin: 0.25em 0 0 0 !important;  /* Reduced top and bottom margins */
            width: 100% !important;
        }}
    </style>
""")

gr.HTML(f"""
    <style>
        /* Conversation List and Manage Bots Button Spacing */
        .conversations-container {{
            display: flex;
            flex-direction: column;
            gap: 2px !important;  /* Reduced gap between elements */
        }}
        #conversations_list {{
            margin-bottom: 0 !important;  /* Remove bottom margin */
        }}
        #manage_bots_btn {{
            order: -1;  /* Move Manage Bots button to the top */
            margin-top: 0 !important;  /* Reset top margin */
            margin-bottom: 5px !important;  /* Add some space below */
            padding: 6px 10px !important;  /* Reduce button padding */
            width: 100% !important;
        }}
        #new_chat_btn {{
            order: 0;  /* Keep New Chat button in its original order */
            margin-top: 0 !important;  /* Reset top margin */
        }}
    </style>
""")
