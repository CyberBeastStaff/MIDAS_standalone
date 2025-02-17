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

import gradio as gr
import ollama
from flask import Flask
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize Flask app
app = Flask(__name__)

# Constants
DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant. Answer questions accurately and concisely."""

# Configure Ollama client
ollama_client = ollama.Client(host='http://localhost:11434')

# Database setup
def init_db():
    """
    Initialize the database with required tables
    """
    try:
        # Connect to the database
        conn = sqlite3.connect('conversations.db')
        cursor = conn.cursor()
        
        # Create conversations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT DEFAULT 'New Conversation',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_used_model TEXT,
                is_active INTEGER DEFAULT 1
            )
        ''')
        
        # Create messages table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER,
                model TEXT,
                role TEXT,
                content TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(conversation_id) REFERENCES conversations(id)
            )
        ''')
        
        # Create documents table for RAG
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                original_name TEXT,
                file_type TEXT,
                embedding_path TEXT,
                conversation_id INTEGER,
                uploaded_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(conversation_id) REFERENCES conversations(id)
            )
        ''')
        
        # Create settings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        ''')
        
        # Check if default model exists
        cursor.execute('SELECT value FROM settings WHERE key = "default_model"')
        result = cursor.fetchone()
        
        if not result:
            # Set default model if not exists
            cursor.execute('''
                INSERT INTO settings (key, value) 
                VALUES ('default_model', 'llama3.1:8b')
            ''')
            print("Initialized default model setting to llama3.1:8b")
        else:
            print(f"Using existing default model: {result[0]}")
        
        # Commit changes and close connection
        conn.commit()
    except Exception as e:
        print(f"Error initializing database: {e}")
        traceback.print_exc()
    finally:
        conn.close()

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
    Get list of available models from Ollama
    
    Returns:
        list: List of model names with their versions
    """
    try:
        # Get models using ollama list command
        models = []
        response = requests.get('http://localhost:11434/api/tags')
        if response.status_code == 200:
            data = response.json()
            if 'models' in data:
                # Keep the full model name with version
                models = [model['name'] for model in data['models']]
        
        # If no models found, use default list based on what's installed
        if not models:
            models = ["phi3.5:latest", "mistral:latest", "llama3.1:8b", "codellama:latest"]
        
        return models
        
    except Exception as e:
        print(f"Error getting models: {e}")
        # Return default models if API fails
        return ["phi3.5:latest", "mistral:latest", "llama3.1:8b", "codellama:latest"]

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
        model_name (str): Model name or full model name
    
    Returns:
        str: Full model name with correct version
    """
    # Model-specific version mapping
    version_map = {
        "llama3.1": "8b",
        "phi3.5": "latest",
        "mistral": "latest",
        "codellama": "latest"
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
        model (str): Model or bot name to use
        chat_history (list): Conversation history
        conv_id (str): Conversation ID
        context (str, optional): RAG context
    
    Returns:
        tuple: Empty string, updated chat history, conversation ID
    """
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
            max_tokens = 4096
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
                "content": f"Here is some relevant context to help answer the user's question, make sure to adhere to the system prompt while using the context:\n\n{combined_context}"
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
                if chunk['message']['content']:
                    response += chunk['message']['content']
                    response_lines.append(chunk['message']['content'])
                    # Update chat history with current response
                    updated_history = chat_history + [(message, response)]
                    yield "", updated_history, conv_id
        except Exception as e:
            print(f"Error during response generation: {e}")
            error_msg = "I apologize, but I encountered an error while generating the response. Please try again."
            updated_history = chat_history + [(message, error_msg)]
            yield "", updated_history, conv_id
            return
        
        # Save final message pair
        save_message(conv_id, model, "user", message)
        save_message(conv_id, model, "assistant", response)
        
        # Generate title for first message
        if is_first_message:
            title = generate_conversation_title(message, response, base_model)
            update_conversation_title(conv_id, title)
        
        # Update conversation model if using a bot
        if bot_config:
            update_conversation_model(model, conv_id)
        
        updated_history = chat_history + [(message, response)]
        yield "", updated_history, conv_id
        
    except Exception as e:
        print(f"Error generating response: {e}")
        traceback.print_exc()
        error_msg = "I apologize, but I encountered an error while processing your request."
        updated_history = chat_history + [(message, error_msg)]
        yield "", updated_history, conv_id

def save_message(conv_id, model, role, content):
    """
    Save a message to the conversation database
    
    Args:
        conv_id (int): Conversation ID
        model (str): Model used
        role (str): Role of the message (user/assistant)
        content (str): Message content
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
    conn = sqlite3.connect('conversations.db')
    c = conn.cursor()
    c.execute('''SELECT content FROM messages 
                 WHERE model = ? ORDER BY timestamp''', (model_name,))
    history = c.fetchall()
    conn.close()
    return history

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
        model (str): Model used for generation
    
    Returns:
        str: Generated title
    """
    try:
        # Special handling for image generation conversations
        if model == "SDXL" or "image" in message.lower():
            import re
            
            # Clean and normalize the message
            clean_message = message.lower().strip()
            clean_message = re.sub(r'^(generate|create|make)\s*(an?\s*)?image\s*(of|with)?', '', clean_message).strip()
            clean_message = re.sub(r'--\w+\s*[^\s]+', '', clean_message).strip()
            
            # Use the default model to generate a more descriptive title
            default_model = get_default_model()
            
            # Prepare messages for title generation
            messages = [
                {
                    "role": "system",
                    "content": """You are an expert at creating concise, informative titles. 
                    Given an image generation prompt, create a title that:
                    1. Captures the essence of the image
                    2. Is no more than 5 words long
                    3. Highlights the most important or unique aspect
                    4. Uses descriptive and engaging language
                    
                    Return ONLY the title, nothing else."""
                },
                {
                    "role": "user",
                    "content": f"Image generation prompt: {clean_message}"
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
                    "num_predict": 50
                }
            ):
                if chunk['message']['content']:
                    title += chunk['message']['content']
            
            # Clean up title
            title = title.strip().strip('"').strip().title()
            
            # Fallback and truncate
            if not title:
                # Extract key words if AI fails
                words = clean_message.split()
                title = ' '.join(words[:3]).title()
            
            # Ensure title is not too long
            return title[:40] if title else "Image Generation"
        
        # Use the model to generate a title for other conversations
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Generate a brief, descriptive title (maximum 6 words) for a conversation based on the user's message and your response. Return ONLY the title, nothing else."
            },
            {
                "role": "user",
                "content": f"User's message: {message}\nYour response: {response}"
            }
        ]
        
        title = ""
        for chunk in ollama_client.chat(
            model=get_model_version(model),
            messages=messages,
            stream=True,
            options={
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": 50
            }
        ):
            if chunk['message']['content']:
                title += chunk['message']['content']
        
        # Clean up title
        title = title.strip().strip('"').strip()
        if not title:
            title = "Untitled Conversation"
        
        return title
        
    except Exception as e:
        print(f"Error generating conversation title: {e}")
        traceback.print_exc()
        return "Untitled Conversation"

def update_conversation_title(conv_id, title):
    """
    Update the title of a conversation
    
    Args:
        conv_id (int): Conversation ID
        title (str): New title
    """
    try:
        conn = sqlite3.connect('conversations.db')
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE conversations
            SET title = ?
            WHERE id = ?
        ''', (title, conv_id))
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
                    match = re.search(r'!\[Generated Image\]\((file/)?(.+)\)', content)
                    if match:
                        image_path = match.group(2)
                        # Remove any 'file/' prefix if it exists and add it back
                        image_path = image_path.replace('file/', '')
                        content = f"![Generated Image](file/{image_path})"
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
                match = re.search(r'!\[Generated Image\]\((file/)?(.+)\)', content)
                if match:
                    image_path = match.group(2)
                    # Remove any 'file/' prefix if it exists and add it back
                    image_path = image_path.replace('file/', '')
                    content = f"![Generated Image](file/{image_path})"
            
            messages.append({
                'role': role,
                'content': content,
                'timestamp': timestamp
            })
        
        # If we detected generated images, force the model to SDXL
        if has_generated_image:
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
            
            # If this is an image generation conversation, force SDXL
            if has_images:
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
    
    Returns:
        pandas.DataFrame: Dataframe of existing bots
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

def load_existing_bots():
    """
    Load existing bot configurations for display
    
    Returns:
        pandas.DataFrame: DataFrame of existing bot configurations
    """
    try:
        bots = get_bot_configurations()
        if not bots:
            return pd.DataFrame(columns=['Name', 'Base Model', 'System Prompt', 'Temperature', 'Max Tokens', 'Top P'])
        
        return pd.DataFrame(bots)
    except Exception as e:
        print(f"Error loading existing bots: {e}")
        return pd.DataFrame(columns=['Name', 'Base Model', 'System Prompt', 'Temperature', 'Max Tokens', 'Top P'])

def get_existing_bots():
    """
    Retrieve list of existing bot names
    
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

def handle_bot_selection(bot_name_dropdown):
    """
    Handle bot selection and load its configuration
    
    Args:
        bot_name_dropdown (str): Selected bot name
    
    Returns:
        tuple: Updated configuration components
    """
    # First, determine custom name input visibility
    custom_name_input_visibility = gr.update(visible=bot_name_dropdown == 'New Bot')
    
    if bot_name_dropdown == 'New Bot':
        # Reset all fields for a new bot
        return (
            custom_name_input_visibility,
            gr.update(value=get_default_model()),
            gr.update(value=''),
            gr.update(value=0.7),
            gr.update(value=4096),
            gr.update(value=0.9)
        )
    
    # Load existing bot configuration
    bot_config = load_bot_configuration(bot_name_dropdown)
    
    if bot_config:
        return (
            custom_name_input_visibility,
            gr.update(value=bot_config['base_model']),
            gr.update(value=bot_config['system_prompt']),
            gr.update(value=bot_config['temperature']),
            gr.update(value=bot_config['max_tokens']),
            gr.update(value=bot_config['top_p'])
        )
    
    # Fallback if configuration not found
    return (
        custom_name_input_visibility,
        gr.update(value=get_default_model()),
        gr.update(value=''),
        gr.update(value=0.7),
        gr.update(value=4096),
        gr.update(value=0.9)
    )

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
        # Generate unique filename
        file_ext = os.path.splitext(original_name)[1].lower()
        unique_filename = f"{uuid.uuid4()}{file_ext}"
        
        # Create documents directory if it doesn't exist
        docs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "documents")
        os.makedirs(docs_dir, exist_ok=True)
        
        # Copy file to documents directory
        permanent_path = os.path.join(docs_dir, unique_filename)
        shutil.copy2(file_path, permanent_path)
        
        # Load document based on file type
        if file_ext == '.pdf':
            loader = PyPDFLoader(permanent_path)
        elif file_ext == '.txt':
            loader = TextLoader(permanent_path)
        elif file_ext in ['.doc', '.docx']:
            loader = Docx2txtLoader(permanent_path)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        # Load and split the document
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            add_start_index=True,
        )
        chunks = text_splitter.split_documents(documents)
        
        # Create embeddings directory
        embeddings_dir = os.path.join(docs_dir, "embeddings", str(uuid.uuid4()))
        os.makedirs(embeddings_dir, exist_ok=True)
        
        # Initialize embeddings model
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Create and persist vector store
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=embeddings_dir
        )
        vectordb.persist()
        
        # Save document info to database
        conn = sqlite3.connect('conversations.db')
        c = conn.cursor()
        c.execute('''INSERT INTO documents 
                     (filename, original_name, file_type, embedding_path, conversation_id)
                     VALUES (?, ?, ?, ?, ?)''',
                 (unique_filename, original_name, file_ext, embeddings_dir, conversation_id))
        conn.commit()
        conn.close()
        
        return True, f"Successfully processed {original_name}"
        
    except Exception as e:
        print(f"Error processing document: {e}")
        traceback.print_exc()
        return False, f"Error processing document: {str(e)}"

def handle_file_upload(file_obj, conversation_id=None):
    """
    Handle file upload from Gradio interface
    
    Args:
        file_obj: Gradio file object
        conversation_id (int, optional): Current conversation ID
    
    Returns:
        str: Status message
    """
    try:
        if file_obj is None:
            return "No file uploaded"
            
        # Get the original filename from the file path
        original_name = os.path.basename(file_obj)
            
        # Process the uploaded file
        success, message = process_document(
            file_obj,  # Gradio now provides the file path directly
            original_name,
            conversation_id
        )
        
        return message
        
    except Exception as e:
        print(f"Error handling file upload: {e}")
        traceback.print_exc()
        return f"Error uploading file: {str(e)}"

def get_relevant_context(query, conversation_id=None, top_k=3):
    """
    Get relevant context for a query from uploaded documents
    
    Args:
        query (str): User's query
        conversation_id (int, optional): Current conversation ID
        top_k (int): Number of chunks to retrieve
    
    Returns:
        str: Relevant context from documents
    """
    try:
        # Get all documents for the conversation
        conn = sqlite3.connect('conversations.db')
        c = conn.cursor()
        
        if conversation_id:
            c.execute('SELECT embedding_path FROM documents WHERE conversation_id = ?', (conversation_id,))
        else:
            c.execute('SELECT embedding_path FROM documents WHERE conversation_id IS NULL')
            
        embedding_paths = c.fetchall()
        conn.close()
        
        if not embedding_paths:
            return ""
        
        # Initialize embeddings model
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Combine results from all documents
        all_results = []
        for path in embedding_paths:
            if os.path.exists(path[0]):
                vectordb = Chroma(
                    persist_directory=path[0],
                    embedding_function=embeddings
                )
                results = vectordb.similarity_search(query, k=top_k)
                all_results.extend(results)
        
        # Sort by relevance and get top results
        all_results = sorted(all_results, key=lambda x: x.metadata.get('score', 0), reverse=True)[:top_k]
        
        # Format context
        if all_results:
            context = "\n\n".join([doc.page_content for doc in all_results])
            return f"Relevant context from documents:\n{context}\n\n"
        
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
        # Generate unique filename
        file_ext = os.path.splitext(original_name)[1].lower()
        unique_filename = f"{str(uuid.uuid4())}{file_ext}"
        
        # Create documents directory if it doesn't exist
        docs_dir = os.path.join('bot_documents')
        os.makedirs(docs_dir, exist_ok=True)
        
        # Copy file to permanent location
        permanent_path = os.path.join(docs_dir, unique_filename)
        shutil.copy2(file_path, permanent_path)
        
        # Load and process document based on file type
        if file_ext == '.pdf':
            loader = PyPDFLoader(permanent_path)
        elif file_ext == '.docx':
            loader = Docx2txtLoader(permanent_path)
        elif file_ext == '.txt':
            loader = TextLoader(permanent_path)
        else:
            os.remove(permanent_path)
            return False, f"Unsupported file type: {file_ext}"
        
        # Load document and split into chunks
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        
        # Generate embeddings
        embeddings_dir = os.path.join('bot_embeddings')
        os.makedirs(embeddings_dir, exist_ok=True)
        embedding_path = os.path.join(embeddings_dir, f"{os.path.splitext(unique_filename)[0]}")
        
        # Save chunks to vector store
        embedding_function = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        Chroma.from_documents(
            documents=chunks,
            embedding=embedding_function,
            persist_directory=embedding_path
        )
        
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
        if not file_path or not bot_name_dropdown or bot_name_dropdown == 'New Bot':
            return "Please select a bot before uploading documents"
        
        # Get bot ID
        conn = sqlite3.connect('bots.db')
        cursor = conn.cursor()
        cursor.execute('SELECT id FROM bots WHERE name = ?', (bot_name_dropdown,))
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return "Bot not found"
        
        bot_id = result[0]
        success, message = process_bot_document(file_path, os.path.basename(file_path), bot_id)
        return message
        
    except Exception as e:
        print(f"Error handling file upload: {e}")
        traceback.print_exc()
        return f"Error uploading file: {str(e)}"

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
        
        # Combine results from all document embeddings
        embedding_function = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Initialize vector store
        all_contexts = []
        for path in embedding_paths:
            try:
                db = Chroma(
                    persist_directory=path[0],
                    embedding_function=embedding_function
                )
                results = db.similarity_search(query, k=2)
                for doc in results:
                    all_contexts.append(doc.page_content)
            except Exception as e:
                print(f"Error searching document {path}: {e}")
                continue
        
        return "\n\n".join(all_contexts) if all_contexts else ""
        
    except Exception as e:
        print(f"Error getting bot knowledge context: {e}")
        return ""

def load_existing_bots():
    """
    Load existing bots from the database into a DataFrame
    
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

def initialize_comfyui():
    """Initialize ComfyUI and set up the global instance"""
    try:
        from comfyui_integration import ComfyUIIntegrator
        global comfyui
        
        print("Starting ComfyUI initialization...")
        comfyui = ComfyUIIntegrator()
        print("ComfyUI initialized successfully")
        return comfyui
        
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

def generate_image_from_prompt(prompt, width, height):
    """Generate an image using ComfyUI based on the given prompt"""
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
        
        print(f"Generating image with prompt: {prompt}, size: {width}x{height}")
        
        # Convert width and height to integers
        width = int(width)
        height = int(height)
        
        # Generate the image with a longer timeout
        image = comfyui.generate_image(prompt, width=width, height=height, timeout=600)  # 10-minute timeout
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
        margin-top: 0.5em !important;
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
        scrollbar-width: none !important;  /* Firefox */
        -ms-overflow-style: none !important;  /* IE and Edge */
        height: calc(100vh - 380px) !important;
        overflow-y: auto !important;
    }
    #conv_list > div::-webkit-scrollbar {
        display: none !important;  /* Chrome, Safari, Opera */
    }
    .gradio-container {
        max-width: 100% !important;
    }
    .gradio-footer {
        display: none !important;
    }
    footer {
        display: none !important;
    }
    #xeno_logo {
        margin: 0.5em auto !important;
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
    }
    /* Adjust spacing for sidebar elements */
    #new_chat_btn {
        margin: 0.5em 0 !important;
    }
    #search_bar {
        margin: 0.5em 0 !important;
    }
    #manage_bots_btn {
        margin: 1em 0 0.5em 0 !important;
        width: 100% !important;
    }
    #app_footer {
        text-align: center !important;
        width: 100% !important;
        padding: 0.5em 0 !important;
        margin-top: 1em !important;
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
""", title="MIDAS 2.0") as demo:

    current_conversation = gr.State()
    current_model = gr.State()
    
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                # Add Xeno logo at the top
                gr.Image(
                    "xeno.png",
                    show_label=False,
                    container=False,
                    show_download_button=False,
                    height=100,
                    elem_id="xeno_logo"
                )
                
                # New chat button
                new_conv_btn = gr.Button(" New Chat", variant="primary")
                
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
                    row_count=10,
                    interactive=False,
                    wrap=True,
                    column_widths=["100%"]
                )
                
                # Manage Bots button
                manage_bots_btn = gr.Button(" Manage Bots", variant="secondary")
            
        with gr.Column(scale=3):
            # Get available models and format them
            available_models = [get_model_display_name(m) for m in get_available_models()]  # Ollama models
            bot_names = get_bot_names()  # Bot names
            all_models = available_models + bot_names + ["SDXL"]  # Combine all models
            
            model_dropdown = gr.Dropdown(
                choices=all_models,
                value=get_default_model(),  # Use the default model from settings
                label="Model",
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
                    elem_id="chat_title"
                )
                rename_chat_btn = gr.Button("Rename", variant="secondary")
                rename_chat_input = gr.Textbox(
                    label="New Title", 
                    visible=False,  # Hidden by default
                    interactive=True
                )
                confirm_rename_btn = gr.Button(
                    "Confirm", 
                    variant="primary", 
                    visible=False  # Hidden by default
                )

            # Rename button click logic
            def toggle_rename_input():
                return {
                    rename_chat_input: gr.update(visible=True),
                    confirm_rename_btn: gr.update(visible=True)
                }

            rename_chat_btn.click(
                fn=toggle_rename_input,
                outputs=[rename_chat_input, confirm_rename_btn]
            )

            # Confirm rename logic
            def confirm_rename(current_chat_title, new_title):
                if not new_title:
                    return current_chat_title, gr.update(visible=False), gr.update(visible=False), conv_list
                
                # Get conversation ID from title
                conn = sqlite3.connect('conversations.db')
                c = conn.cursor()
                
                # Directly query for the conversation ID by title
                c.execute('''SELECT id FROM conversations 
                             WHERE title = ? 
                             ORDER BY created_at DESC 
                             LIMIT 1''', (current_chat_title,))
                
                conv_result = c.fetchone()
                conn.close()
                
                if not conv_result:
                    print(f"No conversation found with title: {current_chat_title}")
                    return current_chat_title, gr.update(visible=False), gr.update(visible=False), conv_list
                
                # Get the conversation ID
                conv_id = conv_result[0]
                
                # Update the conversation title
                rename_current_chat(conv_id, new_title)
                # Get updated conversation list
                updated_conversations = format_conversation_list()
                return new_title, gr.update(visible=False), gr.update(visible=False), updated_conversations
                
            confirm_rename_btn.click(
                fn=confirm_rename,
                inputs=[chat_title, rename_chat_input],
                outputs=[chat_title, rename_chat_input, confirm_rename_btn, conv_list]
            )

            # Conversation options right under the title
            with gr.Row():
                with gr.Column(scale=1):
                    delete_conv_btn = gr.Button(" Delete Current Conversation", variant="stop")
            
            # Chatbot interface
            chatbot = gr.Chatbot(
                label="Conversation", 
                show_label=False, 
                height=500,
                layout="bubble",
                bubble_full_width=False
            )
            
            # Message input and submit area
            with gr.Row():
                msg_box = gr.Textbox(
                    label="Message", 
                    show_label=False, 
                    container=False,
                    placeholder="Type your message here...",
                    lines=3
                )
                submit_btn = gr.Button("Send", variant="primary")
            
            # File upload component
            with gr.Row():
                file_upload = gr.File(
                    label="Upload Document",
                    file_types=[".pdf", ".txt", ".doc", ".docx"],
                    type="filepath"
                )
                upload_status = gr.Markdown()

            # Hidden components for state management
            current_conversation = gr.State(None)
            current_model = gr.State("llama3.1:8b")

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
                        bot_name_dropdown,  # Bot name dropdown
                        base_model,        # Base model dropdown
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
        
        # Connect the close button
        close_dialog_btn = gr.Button("Close", variant="secondary")
        close_dialog_btn.click(
            fn=lambda: gr.update(visible=False),
            outputs=[manage_bots_dialog]
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

    # Event handlers
    demo.load(
        format_conversation_list, 
        outputs=[conv_list]
    )
    
    new_conv_btn.click(
        lambda: [
            None,  # current_conversation
            [],    # chatbot
            "llama3.1:8b",  # model
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
        
        # Get relevant context for non-SDXL models
        context = get_relevant_context(message, conv_id)
        
        # Check if SDXL image generation is requested
        if model_name == "SDXL":
            try:
                # Parse SDXL commands
                sdxl_params, clean_prompt = parse_sdxl_commands(message)
                
                # Generate image using ComfyUI
                global comfyui
                if comfyui is None:
                    comfyui = ComfyUIIntegrator()
                
                # Apply style if specified
                if sdxl_params.get('style'):
                    clean_prompt = f"{clean_prompt} {sdxl_params['style']}"
                
                # Add user message to history first
                history.append((message, "Generating image..."))
                yield history, "", get_chat_title(conv_id), format_conversation_list()
                
                image = comfyui.generate_image(
                    prompt=clean_prompt,
                    width=sdxl_params.get('width', 1024),
                    height=sdxl_params.get('height', 1024),
                    negative_prompt=sdxl_params.get('negative_prompt', ""),
                    steps=sdxl_params.get('steps', 30),
                    cfg=sdxl_params.get('cfg', 7.0),
                    quality=sdxl_params.get('quality', 1.0),
                    seed=sdxl_params.get('seed')
                )
                
                # Ensure conversations directory exists
                os.makedirs('conversations', exist_ok=True)
                
                # Save the image and create markdown
                os.makedirs(f'conversations/{conv_id}', exist_ok=True)
                image_filename = f'conversations/{conv_id}/image_{uuid.uuid4()}.png'
                image.save(image_filename)
                image_markdown = f"![Generated Image](file/{image_filename})"
                
                # Update history with the actual image
                history[-1] = (message, image_markdown)
                
                # Save messages to database
                save_message(conv_id, model_name, "user", message)
                save_message(conv_id, model_name, "assistant", image_markdown)
                
                # Generate title if it's a new conversation
                if get_chat_title(conv_id) == "Untitled Conversation":
                    title = generate_conversation_title(message, image_markdown, model_name)
                    update_conversation_title(conv_id, title)
                
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
                
                # Generate title if it's a new conversation
                if get_chat_title(conv_id) == "Untitled Conversation":
                    title = generate_conversation_title(message, "", model_name)
                    update_conversation_title(conv_id, title)
                
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
                
                # Save the final response
                save_message(conv_id, model_name, "assistant", full_response)
                
                # Update title if still untitled
                if get_chat_title(conv_id) == "Untitled Conversation":
                    title = generate_conversation_title(message, full_response, model_name)
                    update_conversation_title(conv_id, title)
                
                yield history, "", get_chat_title(conv_id), format_conversation_list()
                
            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                print(error_msg)
                print(traceback.format_exc())
                
                # Update history with error
                if len(history) > 0 and history[-1][0] == message:
                    history[-1] = (message, error_msg)
                else:
                    history.append((message, error_msg))
                
                # Save error messages
                save_message(conv_id, model_name, "user", message)
                save_message(conv_id, model_name, "assistant", error_msg)
                
                yield history, "", get_chat_title(conv_id), format_conversation_list()
    
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
            load_existing_bots(),  # Existing bots list
            gr.update(choices=get_existing_bots() + ['New Bot'], value='New Bot'),  # Bot name dropdown
            gr.update(choices=get_available_models(), value=get_default_model())  # Base model dropdown
        ),
        inputs=None,
        outputs=[
            existing_bots_df,  # Existing bots list
            bot_name_dropdown,  # Bot name dropdown
            base_model         # Base model dropdown
        ]
    )

    file_upload.upload(
        fn=handle_file_upload,
        inputs=[file_upload, current_conversation],
        outputs=[upload_status]
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

    # Image generation handler
    def generate_image_from_prompt(prompt, width, height):
        try:
            image = comfyui.generate_image(prompt, width, height)
            return image
        except Exception as e:
            raise gr.Error(f'Image generation failed: {str(e)}')

    # generate_image_btn.click(
    #     fn=generate_image_from_prompt,
    #     inputs=[image_prompt, image_width, image_height],
    #     outputs=generated_image
    # )

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

if __name__ == "__main__":
    # Initialize databases
    init_db()
    init_bots_db()
    
    # Create required directories
    os.makedirs('bot_documents', exist_ok=True)
    os.makedirs('bot_embeddings', exist_ok=True)
    
    # Start the Gradio interface
    demo.queue().launch(server_name="0.0.0.0", share=False, allowed_paths=["."], favicon_path="favicon.ico")
