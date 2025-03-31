import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
from datetime import datetime
import requests
import json
# added for setting system path
import sys
sys.path.append('/path/to/api') 

from knowledge_base import query_knowledge_base  # Import FAISS search function

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Set Groq API Key from environment variable
GROQ_API_KEY = os.environ['GROQ_API_KEY']

# Shopify API credentials (set these in environment variables in production)
SHOPIFY_STORE_URL = os.environ.get('SHOPIFY_STORE_URL', 'https://your-store-name.myshopify.com/admin/api/2023-10')
SHOPIFY_ACCESS_TOKEN = os.environ.get('SHOPIFY_ACCESS_TOKEN', 'your_access_token_here')

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# Function to fetch order details from Shopify
def get_shopify_order(order_id):
    url = f"{SHOPIFY_STORE_URL}/orders/{order_id}.json"
    headers = {
        "X-Shopify-Access-Token": SHOPIFY_ACCESS_TOKEN,
        "Content-Type": "application/json"
    }
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            order_data = response.json().get("order", {})
            status = order_data.get("fulfillment_status", "Not fulfilled")
            tracking_url = (order_data["fulfillments"][0].get("tracking_url") 
                          if "fulfillments" in order_data and order_data["fulfillments"] 
                          else "No tracking available yet")
            return {
                "status": status,
                "tracking_url": tracking_url,
                "order_number": order_data.get("order_number", order_id)
            }
        else:
            return {"error": f"Order not found (Error: {response.status_code})"}
    except Exception as e:
        return {"error": f"Failed to fetch order: {str(e)}"}

# Define the chatbot's context
CONTEXT = """You are NOVA, a proactive AI assistant for Nexobotics. You provide customer support, track Shopify orders, and answer questions using a knowledge base.

**Knowledge Base:** If a user asks something that seems knowledge-based (e.g., "How does AI improve customer support?"), search the FAISS database and summarize the most relevant result. 

**Shopify Orders:** If a user asks about an order (e.g., "Where’s my order 12345?"), request the order details using the function `[CALL: get_shopify_order(order_id)]`.

Stay concise, engaging, and provide direct answers. Keep greetings short and end conversations with motivational or engaging lines.
"""

# Initialize chat history for each session
chat_histories = {}

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    if not request.is_json:
        return jsonify({'error': 'Content-Type must be application/json'}), 400

    message = request.json.get('message')
    session_id = request.json.get('session_id', str(datetime.now()))  # Default session ID

    if not message:
        return jsonify({'error': 'Message is required'}), 400

    try:
        # Initialize or retrieve chat history for the session
        if session_id not in chat_histories:
            chat_histories[session_id] = [{"role": "system", "content": CONTEXT}]

        # Add user prompt to history
        chat_histories[session_id].append({"role": "user", "content": message})

        # **Check for Knowledge Base Query**
        faiss_result = query_knowledge_base(message)  # Query FAISS database
        if faiss_result:
            chat_histories[session_id].append({"role": "system", "content": f"Knowledge Base Info: {faiss_result}"})

        # Generate initial response
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=chat_histories[session_id],
            temperature=0.8,
            max_tokens=1024
        )

        ai_response = response.choices[0].message.content
        chat_histories[session_id].append({"role": "assistant", "content": ai_response})

        # **Check if NOVA requested an order lookup**
        if "[CALL: get_shopify_order(" in ai_response:
            try:
                start = ai_response.index("[CALL: get_shopify_order(") + 25
                end = ai_response.index(")]", start)
                order_id = ai_response[start:end].strip("'\"")

                order_info = get_shopify_order(order_id)
                chat_histories[session_id].append({
                    "role": "system",
                    "content": f"Order info for {order_id}: {json.dumps(order_info)}"
                })

                # Generate a follow-up response with order details
                follow_up = client.chat.completions.create(
                    model="llama3-70b-8192",
                    messages=chat_histories[session_id],
                    temperature=0.8,
                    max_tokens=1024
                )
                ai_response = follow_up.choices[0].message.content
                chat_histories[session_id].append({"role": "assistant", "content": ai_response})
            except Exception as e:
                ai_response = f"Oops, something went wrong while checking that order: {str(e)}"

        return jsonify({'response': ai_response})
    except Exception as e:
        print(f"Error processing message: {str(e)}")  # For debugging
        return jsonify({'error': 'An error occurred processing your request'}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))  # Render uses the PORT environment variable
    app.run(host='0.0.0.0', port=port)
