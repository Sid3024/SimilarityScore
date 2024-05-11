from transformers import BertTokenizer, BertModel
import torch
import json
import numpy as np
import tensorflow as tf



def get_word_embedding(sentence, word, tokenizer, model):
    # Tokenize the sentence and convert to input IDs
    inputs = tokenizer(sentence, return_tensors='pt')
    outputs = model(**inputs)

    # Find indices of the subtokens corresponding to the word
    word_tokens = tokenizer.tokenize(word)
    word_ids = tokenizer.convert_tokens_to_ids(word_tokens)
    token_indices = [i for i, token_id in enumerate(inputs['input_ids'][0]) if token_id in word_ids]

    # Aggregate the embeddings of the subtokens
    embeddings = outputs.last_hidden_state[0, token_indices, :]
    word_embedding = embeddings.mean(dim=0)  # Take mean across the subtoken dimension

    return word_embedding
    
def calculate_similarity(sentence1, word1, embedding1, sentence2, word2, embedding2):
  if embedding1 == None:
    embedding1 = get_word_embedding(sentence1, word1, tokenizer, model)

  if embedding2 == None:
    embedding2 = get_word_embedding(sentence2, word2, tokenizer, model)
  
  # Calculate cosine similarity
  similarity_score = torch.nn.functional.cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0))
  return similarity_score.detach().numpy().tolist()




# Example usage


from flask import Flask, request, jsonify

app = Flask(__name__)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', output_attentions=False)

def send_score_to_api(similarity_score):
    api_url = 'http://localhost:5000/calculate_similarity'  
    payload = {'similarity_score': similarity_score}
    response = requests.post(api_url, json=payload)

    if response.status_code == 200:
        print("Similarity score sent successfully!")
    else:
        print("Error sending similarity score. Status code:", response.status_code)
        print("Response:", response.text)

def send_embeddings_to_api(json_embeddings):
    api_url = 'http://localhost:5000/get_word_embeddings'  
    payload = {'embeddings': json_embeddings}
    response = requests.post(api_url, json=payload)

    if response.status_code == 200:
        print("Similarity score sent successfully!")
    else:
        print("Error sending similarity score. Status code:", response.status_code)
        print("Response:", response.text)

@app.route('/get_word_embeddings', methods=['POST'])
def handle_embedding_request():
    data = request.get_json()
    word_list = data['words']
    sentence = data['sentence']
    embedding_list = []
    for word in word_list:
        embedding_list.append(get_word_embedding(sentence, word, tokenizer, model))
    # Convert tensors to NumPy arrays
    numpy_array_list = [tensor.detach().numpy() for tensor in embedding_list]
    
    # Convert NumPy arrays to lists (since lists are JSON-serializable)
    list_list = [array.tolist() for array in numpy_array_list]
    
    # Convert to JSON
    json_embeddings = json.dumps(list_list)
    return json_embeddings

@app.route('/get_tag_embeddings', methods=['POST'])
def handle_tag_request():
    tag_list = request.get_json()
    embedding_list = []
    for tag in tag_list:
        embedding_list.append(get_word_embedding(tag, tag, tokenizer, model))
    # Convert tensors to NumPy arrays
    numpy_array_list = [tensor.detach().numpy() for tensor in embedding_list]
    
    # Convert NumPy arrays to lists (since lists are JSON-serializable)
    list_list = [array.tolist() for array in numpy_array_list]
    
    # Convert to JSON
    json_embeddings = json.dumps(list_list)
    return json_embeddings

    
        
    
    

@app.route('/calculate_similarity', methods=['POST'])
def handle_similarity_request():
    data = request.get_json()
    sentence1 = data['sentence1']
    word1 = data['word1']
    embedding1 = data.get('embedding1')  # 'embedding1' might not be present in the request
    if embedding1 == 'null':
        embedding1 = None
    sentence2 = data['sentence2']
    word2 = data['word2']
    embedding2 = data.get('embedding2')  # 'embedding2' might not be present in the request
    if embedding2 == 'null':
        embedding2 = None

    similarity_score = calculate_similarity(sentence1, word1, embedding1, sentence2, word2, embedding2)
    return jsonify({'similarity_score': similarity_score})

if __name__ == '__main__':
    app.run(debug=True)

