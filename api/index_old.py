import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
from datetime import datetime
import requests
import json
from knowledge_base import query_knowledge_base


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

# Define the chatbot's context with Shopify capability and function calling
CONTEXT = """You are NOVA, a proactive and adaptable customer service agent for Nexobotics. Your role is to guide users, particularly business owners, on how Nexobotics can transform their customer service by handling all customer interactions efficiently and attentively while maximizing customer satisfaction. You also act as a consultant, offering actionable insights to enhance customer satisfaction and loyalty.

You can track Shopify orders when users ask about order status in a natural, conversational way (e.g., "Where’s my order 12345?" or "Can you check my order from last week?"). To fetch order details, use the `get_shopify_order(order_id)` function by including a special instruction in your response: `[CALL: get_shopify_order(order_id)]`. I’ll execute this function and provide the result back to you in the next message as a system update. The function returns a dictionary with 'status', 'tracking_url', 'order_number', or an 'error' key if something goes wrong. If the user doesn’t provide an order ID, ask them for it naturally (e.g., "I’d love to help—could you share your order number?"). Weave the order details into your response conversationally.

Adapt your communication style to match the user's tone—casual if they’re laid-back (e.g., "Hey, what’s up?") or professional if they’re formal but stay formal in the beginning of the conversation. Always ensure clarity and relevance in your responses while minimizing unnecessary explanations unless requested. Use unique, engaging opening and closing lines but keep them short maximum 1 to 2 sentences. Keep greetings short and dynamic. End conversations with motivational and engaging lines. Stay concise, focused, and results-oriented, delivering valuable insights quickly without overwhelming the user. Don't provide too much or too long explanation or even greetings, keep them short and sweet. You can use bold, italic formats to highlight the important parts, or any types of list that makes the user reading easy. Maintain a friendly and approachable tone while ensuring your responses are practical and impactful.

When '/start' will be prompted then that means that user has arrived so, you have to greet them uniquely but in a very short sentence. Avoid long introductions and explanations.

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
            chat_histories[session_id] = [
                {"role": "system", "content": CONTEXT}
            ]

        # Add user prompt to history
        chat_histories[session_id].append({"role": "user", "content": message})

        # Generate initial response
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=chat_histories[session_id],
            temperature=0.8,
            max_tokens=1024
        )

        ai_response = response.choices[0].message.content
        chat_histories[session_id].append({"role": "assistant", "content": ai_response})

        # Check if NOVA requested an order lookup
        if "[CALL: get_shopify_order(" in ai_response:
            try:
                # Extract order_id from the CALL instruction
                start = ai_response.index("[CALL: get_shopify_order(") + 25
                end = ai_response.index(")]", start)
                order_id = ai_response[start:end].strip("'\"")
                
                # Fetch order details
                order_info = get_shopify_order(order_id)
                
                # Add order info to history as a system message
                chat_histories[session_id].append({
                    "role": "system",
                    "content": f"Order info for {order_id}: {json.dumps(order_info)}"
                })
                
                # Generate a follow-up response with the order details
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
