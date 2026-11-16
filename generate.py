import chromadb
import pandas as pd
import networkx as nx
import ollama

def setup_retrieval():
    data_path = r"D:\NeuroGraph_Project\Data\cleaned_disease_symptoms.csv"
    db_path = r"D:\NeuroGraph_Project\Chroma_DB"

    df = pd.read_csv(data_path)
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(name="symptom_collection")

    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(row['Disease'], row['Symptom'], relationship="HAS_SYMPTOM")

    return collection, G

def retrieve_context(user_query, collection, G):
    results = collection.query(query_texts=[user_query], n_results=3)
    matched_symptoms = results['documents'][0]

    candidate_diseases = {}
    for symp in matched_symptoms:
        if G.has_node(symp):
            for disease in G.predecessors(symp):
                candidate_diseases[disease] = candidate_diseases.get(disease, 0) + 1

    sorted_candidates = sorted(candidate_diseases.items(), key=lambda x: x[1], reverse=True)[:3]
    return matched_symptoms, sorted_candidates

def generate_medical_response(user_query, matched_symptoms, candidates):
    context_str = f"Mapped Symptoms: {', '.join(matched_symptoms)}\n"
    context_str += "Candidate Diseases (Ranked by Evidence):\n"
    for disease, count in candidates:
        context_str += f"- {disease.title()} (Matches {count} symptoms)\n"

    system_prompt = (
        "You are an AI clinical explanation engine in a NeuroGraph RAG architecture. "
        "Strict Constraint: Rely strictly on the provided graph context. "
        "Do not invent facts, symptoms, or diagnoses outside the provided information. "
        "Provide a concise, professional diagnostic breakdown."
    )

    user_prompt = f"Patient Input: \"{user_query}\"\n\nRetrieved Knowledge:\n{context_str}\nProvide clinical reasoning:"

    response = ollama.chat(
        model="llama3.2",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response['message']['content']

if __name__ == "__main__":
    col, graph = setup_retrieval()
    
    query = "I have a sudden high fever with severe joint pain and chills."
    print(f"\nEvaluating Query: '{query}'")
    
    symptoms, top_diseases = retrieve_context(query, col, graph)
    explanation = generate_medical_response(query, symptoms, top_diseases)
    
    print("\n--- NeuroGraph RAG Clinical Explanation ---\n")
    print(explanation)