import chromadb
import pandas as pd
import networkx as nx
import numpy as np

def load_resources():
    data_path = r"D:\NeuroGraph_Project\Data\cleaned_disease_symptoms.csv"
    db_path = r"D:\NeuroGraph_Project\Chroma_DB"
    df = pd.read_csv(data_path)
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(name="symptom_collection")
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(row['Disease'], row['Symptom'], relationship="HAS_SYMPTOM")
    return df, collection, G

def standard_vector_rag(query_symptoms, collection, df, k=3):
    matched_diseases = []
    for q in query_symptoms:
        res = collection.query(query_texts=[q], n_results=1)
        if res['documents'][0]:
            symp = res['documents'][0][0]
            matching_diseases = df[df['Symptom'] == symp]['Disease'].tolist()
            if matching_diseases:
                disease = matching_diseases[0]
                if disease not in matched_diseases:
                    matched_diseases.append(disease)
    if len(matched_diseases) < k:
        popular = df['Disease'].value_counts().index.tolist()
        for d in popular:
            if d not in matched_diseases:
                matched_diseases.append(d)
            if len(matched_diseases) >= k:
                break
    return matched_diseases[:k]

def neurograph_hybrid_rag(query_symptoms, collection, G, k=3):
    resolved_symptoms = []
    for q in query_symptoms:
        res = collection.query(query_texts=[q], n_results=1)
        if res['documents'][0]:
            resolved_symptoms.append(res['documents'][0][0])
    candidate_scores = {}
    for symp in resolved_symptoms:
        if G.has_node(symp):
            for disease in G.predecessors(symp):
                candidate_scores[disease] = candidate_scores.get(disease, 0) + 1
    sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:k]
    return [item[0] for item in sorted_candidates]

def calculate_metrics(ground_truth_disease, retrieved_candidates, k=3):
    hit_at_k = 1 if ground_truth_disease in retrieved_candidates[:k] else 0
    accuracy = 1.0 if len(retrieved_candidates) > 0 and ground_truth_disease == retrieved_candidates[0] else 0.0
    
    precision = (1 / k) if hit_at_k else 0.0
    recall = 1.0 if hit_at_k else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    
    # Real Clinical Zero-Hallucination Logic:
    # Agar system galat bimari uthayega (Hit@3 = 0), toh LLM galat treatment hallucinate karega.
    zero_hallucination = 1.0 if hit_at_k == 1 else 0.0
    
    return hit_at_k, precision, recall, f1, accuracy, zero_hallucination
def run_benchmark():
    df, collection, G = load_resources()
    disease_grouped = df.groupby('Disease')['Symptom'].apply(list).to_dict()
    test_cases = list(disease_grouped.items())[:25] # Testing on 25 diseases
    
    print(f"\nNEUROGRAPH RAG BENCHMARK EVALUATION (N={len(test_cases)})")
    
    base_m = {"hit@3": [], "precision": [], "recall": [], "f1": [], "accuracy": [], "zero_halluc": []}
    hyb_m = {"hit@3": [], "precision": [], "recall": [], "f1": [], "accuracy": [], "zero_halluc": []}
    
    for true_disease, symptoms in test_cases:
        query_input = symptoms[:min(4, len(symptoms))]
        
        # Base RAG
        base_preds = standard_vector_rag(query_input, collection, df, k=3)
        b_h, b_p, b_r, b_f1, b_acc, b_zh = calculate_metrics(true_disease, base_preds, k=3)
        base_m["hit@3"].append(b_h); base_m["precision"].append(b_p); base_m["recall"].append(b_r)
        base_m["f1"].append(b_f1); base_m["accuracy"].append(b_acc); base_m["zero_halluc"].append(b_zh)
        
        # Hybrid RAG
        hyb_preds = neurograph_hybrid_rag(query_input, collection, G, k=3)
        h_h, h_p, h_r, h_f1, h_acc, h_zh = calculate_metrics(true_disease, hyb_preds, k=3)
        hyb_m["hit@3"].append(h_h); hyb_m["precision"].append(h_p); hyb_m["recall"].append(h_r)
        hyb_m["f1"].append(h_f1); hyb_m["accuracy"].append(h_acc); hyb_m["zero_halluc"].append(h_zh)
        
    print("-" * 68)
    print(f"{'Metric':<20} | {'Standard Vector RAG':<20} | {'NeuroGraph Hybrid RAG':<20}")
    print("-" * 68)
    
    metrics_names = []
    base_scores = []
    hyb_scores = []

    for m in ["hit@3", "precision", "recall", "f1", "accuracy", "zero_halluc"]:
        base_avg = np.mean(base_m[m]) * 100 
        hyb_avg = np.mean(hyb_m[m]) * 100
        print(f"{m.upper():<20} | {base_avg:<19.2f}% | {hyb_avg:<19.2f}%")
        metrics_names.append(m.upper())
        base_scores.append(base_avg)
        hyb_scores.append(hyb_avg)

    results_df = pd.DataFrame({
        "Metric": metrics_names,
        "Standard Vector RAG": base_scores,
        "NeuroGraph Hybrid RAG": hyb_scores
    })
    
    output_path = r"D:\NeuroGraph_Project\Data\metrics_results.csv" 
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    run_benchmark()