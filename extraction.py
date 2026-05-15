import json
import pandas as pd

def extract_to_csv(file_path, output_filename="extractions_report.csv"):
    # On utilise file_path ici pour ouvrir le fichier
    with open(file_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
        
    rows = []
    for entry in json_data:
        claim_id = entry.get("news_id")
        claim_title = entry.get("news")
        
        database = entry.get("database", [])
        for index, target in enumerate(database):
            rows.append({
                "claim_id": claim_id,
                "claim_title": claim_title,
                "target_label": f"T{index + 1}",
                "target_title": target.get("news"),
                "similarity_score": target.get("similarity_annotation"),
                "relation": target.get("related")
            })
            
    df = pd.DataFrame(rows)
    df.to_csv(output_filename, index=False, encoding='utf-8')
    print(f"Fichier {output_filename} enregistré avec succès.")
    return df