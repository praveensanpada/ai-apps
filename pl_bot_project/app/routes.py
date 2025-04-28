from flask import Flask, request, jsonify
from app.grammar_correction import correct_grammar
from app.nlp import answer_question
from app.retrieval import retrieve_answer
import json

app = Flask(__name__)

# Load the FAQ data (training data in JSON format)
with open('knowledge_base/faq_data.json', 'r') as file:
    faq_data = json.load(file)

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.json['question']
    
    # Correct user grammar if necessary
    corrected_input = correct_grammar(user_input)
    
    # Use embeddings for relevant answers
    context_index = retrieve_answer(corrected_input)
    context = faq_data[context_index[0]]  # Retrieve most relevant context based on similarity
    
    # Use the context for answering the question
    answer = answer_question(corrected_input, context['answer'])
    
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)
