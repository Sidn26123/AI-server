import torch
import ollama
import json
from openai import OpenAI
from .models import Document

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

def get_embeddings_cache():
    """Get or create embeddings for all documents in the database"""
    import pickle
    import os
    
    cache_file = "embeddings_cache.pkl"
    
    # Try to load from cache
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                cache = pickle.load(f)
                return cache['embeddings'], cache['document_ids']
        except:
            pass  # If loading fails, regenerate embeddings
    
    # Generate new embeddings
    documents = Document.objects.all()
    embeddings = []
    document_ids = []
    
    for doc in documents:
        response = ollama.embeddings(model='llama3.2:1b', prompt=doc.content)
        embeddings.append(response["embedding"])
        document_ids.append(doc.id)
    
    embeddings_tensor = torch.tensor(embeddings) if embeddings else torch.tensor([])
    
    # Save to cache
    with open(cache_file, 'wb') as f:
        pickle.dump({
            'embeddings': embeddings_tensor,
            'document_ids': document_ids
        }, f)
    
    return embeddings_tensor, document_ids

def get_relevant_context(rewritten_input, top_k=3):
    embeddings_tensor, document_ids = get_embeddings_cache()
    
    if embeddings_tensor.nelement() == 0:  # Check if the tensor has any elements
        return []
    
    # Encode the rewritten input
    input_embedding = ollama.embeddings(model='llama3.2:1b', prompt=rewritten_input)["embedding"]
    
    # Compute cosine similarity
    cos_scores = torch.cosine_similarity(torch.tensor(input_embedding).unsqueeze(0), embeddings_tensor)
    
    # Adjust top_k if needed
    top_k = min(top_k, len(cos_scores))
    
    # Get top indices
    top_indices = torch.topk(cos_scores, k=top_k)[1].tolist()
    
    # Get relevant documents
    relevant_docs = []
    for idx in top_indices:
        doc_id = document_ids[idx]
        doc = Document.objects.get(id=doc_id)
        relevant_docs.append(doc.content.strip())
    
    return relevant_docs

def rewrite_query(user_input, conversation_history, ollama_model="llama3.2:1b"):
    client = OpenAI(
        base_url='http://localhost:11434/v1',
        api_key=ollama_model
    )
    
    # Extract last 2 messages for context
    recent_messages = conversation_history[-2:] if len(conversation_history) >= 2 else conversation_history
    context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_messages])
    
    prompt = f"""Rewrite the following query by incorporating relevant context from the conversation history.
    The rewritten query should:
    
    - Preserve the core intent and meaning of the original query
    - Expand and clarify the query to make it more specific and informative for retrieving relevant context
    - Avoid introducing new topics or queries that deviate from the original query
    - DONT EVER ANSWER the Original query, but instead focus on rephrasing and expanding it into a new query
    
    Return ONLY the rewritten query text, without any additional formatting or explanations.
    
    Conversation History:
    {context}
    
    Original query: [{user_input}]
    
    Rewritten query: 
    """
    
    response = client.chat.completions.create(
        model=ollama_model,
        messages=[{"role": "system", "content": prompt}],
        max_tokens=200,
        n=1,
        temperature=0.1,
    )
    
    rewritten_query = response.choices[0].message.content.strip()
    return rewritten_query

def ollama_chat(user_input, system_message, conversation_history, ollama_model="llama3.2:1b"):
    client = OpenAI(
        base_url='http://localhost:11434/v1',
        api_key=ollama_model
    )
    
    # Add user input to conversation history
    conversation_history.append({"role": "user", "content": user_input})
    
    # Rewrite query if we have conversation history
    if len(conversation_history) > 1:
        rewritten_query = rewrite_query(user_input, conversation_history, ollama_model)
    else:
        rewritten_query = user_input
    
    # Get relevant context
    relevant_context = get_relevant_context(rewritten_query)
    
    # Prepare user input with context
    user_input_with_context = user_input
    if relevant_context:
        context_str = "\n".join(relevant_context)
        user_input_with_context = user_input + "\n\nRelevant Context:\n" + context_str
    
    # Update the last user message with context
    conversation_history[-1]["content"] = user_input_with_context
    
    # Prepare messages for LLM
    messages = [
        {"role": "system", "content": system_message},
        *conversation_history
    ]
    
    # Get response from LLM
    response = client.chat.completions.create(
        model=ollama_model,
        messages=messages,
        max_tokens=2000,
    )
    
    # Add assistant response to conversation history
    assistant_response = response.choices[0].message.content
    conversation_history.append({"role": "assistant", "content": assistant_response})
    
    return {
        "original_query": user_input,
        "rewritten_query": rewritten_query,
        "relevant_context": relevant_context if relevant_context else [],
        "response": assistant_response,
        "conversation_history": conversation_history
    }