from transformers import BertTokenizer, BertModel
import torch


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



# Example usage
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', output_attentions=False)

sentence1 = "box of sneakers"
word1 = "sneakers"
embedding1 = get_word_embedding(sentence1, word1, tokenizer, model)

sentence2 = "red fruits"
word2 = "fruits"
embedding2 = get_word_embedding(sentence2, word2, tokenizer, model)

# Calculate cosine similarity
cosine_sim = torch.nn.functional.cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0))
print(f"Cosine similarity: {cosine_sim.item()}")
