import torch
from indexing_copali import processor,device,model




def retrieve_top_k(query: str, page_embeddings: list[torch.Tensor], top_k: int = 2):
    # Process query
    with torch.no_grad():
        inputs = processor.process_queries([query]).to(device)
        query_embeddings = model(**inputs)  # Shape: (1, query_tokens, embed_dim)

    # Compute MaxSim scores across all pages
    scores = []
    for page_emb in page_embeddings:
        # Move tensors to appropriate device
        q_emb = query_embeddings[0].to(device)  # (q_tokens, dim)
        p_emb = page_emb.to(device)              # (p_tokens, dim)
        
        # Matrix multiplication for cosine similarities between all tokens/patches
        similarity_matrix = torch.matmul(q_emb, p_emb.T)  # (q_tokens, p_tokens)
        
        # MaxSim: take max along patch dimension, then sum over query tokens
        max_sim_per_query_token, _ = torch.max(similarity_matrix, dim=1)
        score = torch.sum(max_sim_per_query_token).item()
        scores.append(score)

    # Sort indices by descending score
    top_indices = torch.argsort(torch.tensor(scores), descending=True)[:top_k]
    return [indices.item() for indices in top_indices]