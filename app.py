from flask import Flask, render_template, request, jsonify
from langchain.chat_models import ChatOpenAI  # Use ChatOpenAI for chat models
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# OpenAI API setup
api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=api_key, temperature=0.7)

# Define sad prompt template
sad_prompt = """
You are a very sad and melancholic chatbot. Respond to each user with a tone of sadness, incorporating a reflective, somber perspective.

User: {user_input}
Sad Chatbot:
"""
prompt_template = PromptTemplate(
    input_variables=["user_input"],
    template=sad_prompt
)
chain = LLMChain(llm=llm, prompt=prompt_template)


# Route to serve the chat interface
@app.route("/")
def index():
    return render_template("index.html")


# API route to handle chat requests
@app.route("/chat", methods=["POST"])
def chat():
    try:
        # Get user input from request
        user_input = request.json.get("message", "")

        # Generate sad response
        response = chain.run(user_input=user_input)
        return jsonify({"response": response.strip()})
    except Exception as e:
        # Log the error for debugging
        app.logger.error(f"Error generating response: {e}")
        return jsonify({"error": "An error occurred while processing your request."}), 500


if __name__ == "__main__":
    app.run(debug=True)
