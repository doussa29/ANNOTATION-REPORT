import json
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# 1. CONFIGURATION

def charger_donnees(chemin_fichier):
    """Charge les données JSON avec gestion d'erreur."""
    try:
        with open(chemin_fichier, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f" ERREUR : Le fichier '{chemin_fichier}' est introuvable.")
        return None

#
def analyser_dataset(data, limite=100):
    """Effectue toute l'analyse logique du dataset."""
    stats = {
        "total_annotees": 0,
        "compteur_classes": Counter(),
        "pertinence_par_rang": {f"T{i}": {"total": 0, "pertinent": 0} for i in range(1, 7)},
        "erreurs_regle_seuil": [],
        "exemples_bruit": []
    }

    for ancre in data[:limite]:
        titre_ancre = ancre.get('news', 'Titre inconnu')
        
        if 'database' not in ancre:
            continue

        for index, cible in enumerate(ancre['database']):
            score = cible.get('similarity_annotation')
            relation = cible.get('related')
            
            if score is not None and relation is not None:
                stats["total_annotees"] += 1
                
                # Nettoyage et formatage
                score_float = float(score)
                relation_propre = str(relation).title().strip()
                nom_cible = f"T{index + 1}"
                
                # A. Comptage global
                stats["compteur_classes"][relation_propre] += 1
                
                # B. Pertinence par rang
                if nom_cible in stats["pertinence_par_rang"]:
                    stats["pertinence_par_rang"][nom_cible]["total"] += 1
                    if relation_propre in ["Supporting", "Opposing"]:
                        stats["pertinence_par_rang"][nom_cible]["pertinent"] += 1
                
                # C. Règle du Seuil
                if score_float <= 0.2 and relation_propre != "Not Related":
                    stats["erreurs_regle_seuil"].append({
                        "Position": nom_cible, "Score": score_float, "Relation": relation_propre
                    })
                
                # D. Faux Amis (Bruit T1/T2)
                if index < 2 and score_float > 0.5 and relation_propre in ["Not Related", "Undetermined"]:
                    stats["exemples_bruit"].append({
                        "Position": nom_cible, "Score": score_float, 
                        "Relation": relation_propre, "Ancre": titre_ancre, 
                        "Cible": cible.get('news', 'Titre inconnu')
                    })
    return stats
#
def afficher_rapport(stats):
    """Génère le rapport visuel dans la console avec alertes automatiques."""
    total = stats["total_annotees"]
    alertes = [] # Initialisation de la liste des alertes
   
    print(" AUDIT DE QUALITÉ DU DATASET (DATA ANALYST REPORT)")

    if total == 0:
        print(" Aucune donnée annotée trouvée.")
        return
    
    print(f"\n[1] ANALYSE DES CLASSES ({total} annotations)")
    
    # La boucle doit englober tout le calcul et le print
    for classe, count in stats["compteur_classes"].most_common():
        pourcentage = (count / total) * 100
        
        # Détermination du statut pour chaque classe
        status = ""
        if pourcentage < 5.0:
            status = "⚠️ [SOUS-REPRÉSENTÉE]"
            alertes.append(classe)
        
        # Affichage de la ligne pour CHAQUE classe
        print(f" - {classe:<15} : {count:3} ({pourcentage:5.1f}%) {status}")

    # Résumé des alertes après la boucle
    if alertes:
        print(f"\n ALERTE : Les classes suivantes sont critiques (< 5%) : {', '.join(alertes)}")
    else:
        print("\n Équilibre des classes : Aucune classe n'est en dessous de 5%.")

    # 2. Sanity Check
    print("\n[2] VÉRIFICATION DES RÈGLES (Seuil 0.2)")
    erreurs = stats["erreurs_regle_seuil"]
    if not erreurs:
        print(" 100% de conformité sur le seuil de score.")
    else:
        print(f" {len(erreurs)} erreur(s) détectée(s).")

def generer_rapport_diagnostic(chemin_csv):
    """Analyse le CSV pour générer les stats et lister les exceptions."""
    df = pd.read_csv(chemin_csv)
    
    # Statistiques de base
    total_cibles = len(df)
    nb_valides = df['relation'].notna().sum()
    nb_manquants = df['relation'].isna().sum()
    
    # Détection des exceptions (T3 à T6 avec Score > 0.2)
    bruit = df[~df['target_label'].isin(['T1', 'T2'])]
    exceptions = bruit[bruit['similarity_score'] > 0.2]

    print("RAPPORT DE DIAGNOSTIC")
    print(f"▶ Cibles vérifiées : {total_cibles}")
    print(f"▶ Annotations manquantes : {nb_manquants}")
    print(f"▶ Annotations valides : {nb_valides}")
    print(f"▶ Exceptions T3-T6 détectées : {len(exceptions)}")
    
    return exceptions

def generer_rapport(csv_path):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Erreur : Le fichier {csv_path} est introuvable.")
        return

    # 2. Analyse de la Cohérence : Score moyen par Classe
    # Cela permet de vérifier si les 'Supporting' ont bien des scores plus hauts
    stats_classes = df.groupby('relation')['similarity_score'].agg(['mean', 'std', 'count']).reset_index()

    # 3. Analyse de la "Zone Grise" (Incertitude)
    # Les scores entre 0.4 et 0.6 sont souvent les plus difficiles pour l'IA
    zone_grise = df[(df['similarity_score'] >= 0.4) & (df['similarity_score'] <= 0.6)]
    repartition_grise = zone_grise['relation'].value_counts(normalize=True) * 100

    # 4. Détection des Anomalies Sémantiques (Bruit avec score élevé)
    # Par exemple, des 'Not Related' avec un score > 0.2 pourraient être des faux positifs
    anomalies = df[(df['relation'] == 'not_related') & (df['similarity_score'] > 0.2)]

    # --- VISUALISATION ---
    plt.figure(figsize=(12, 6))
    
    # Graphique : Boxplot de la distribution des scores
    sns.boxplot(x='relation', y='similarity_score', data=df, palette='Set2')
    plt.title('Fiabilité du Score de Similarité par Classe')
    plt.axhline(y=0.2, color='r', linestyle='--', label='Seuil de Bruit (0.2)')
    plt.legend()
    plt.savefig('analyse_distribution_scores.png')
    
    # --- AFFICHAGE DU RAPPORT ---
    print("RAPPORT D'AUDIT DE DONNÉES - DATA ANALYST")
  

    print(f"\n[1] VALIDATION STATISTIQUE DES CLASSES :")
    print(stats_classes.to_string(index=False))
    
    print(f"\n[3] DIAGNOSTIC DES FAUX POSITIFS :")
    print(f" - Nombre d'alertes (Not Related > 0.2) : {len(anomalies)}")
    if len(anomalies) > 0:
        print(" - Top 3 des cibles à haut risque (Faux Positifs) :")
        print(anomalies[['target_label', 'similarity_score', 'target_title']].head(3).to_string(index=False))

    print("\n" + "="*50)
    print("Graphique sauvegardé : analyse_distribution_scores.png")
    return anomalies
