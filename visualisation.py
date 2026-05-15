import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



def generer_graphique_similarite(chemin_csv):
    """Affiche le nuage de points des scores à partir du CSV."""
    df = pd.read_csv(chemin_csv)
    # On renomme pour la cohérence des graphiques
    df = df.rename(columns={'target_label': 'Cible', 'similarity_score': 'Score'})
    
    plt.figure(figsize=(11, 5))
    # Zones de couleurs pour distinguer le haut du panier du bruit
    plt.axvspan(-0.5, 1.5, color='#d4edda', alpha=0.4, label='Top Résultats (T1-T2)')
    plt.axvspan(1.5, 5.5, color='#f8d7da', alpha=0.4, label='Bruit (T3-T6)')
    
    sns.stripplot(data=df, x='Cible', y='Score', jitter=True, alpha=0.6, palette='tab10', hue='Cible', legend=False)
    
    plt.title("Distribution des Scores de Similarité", fontweight='bold')
    plt.yticks(np.arange(0.0, 1.1, 0.1))
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig("distribution_scores.png")
    plt.show()

def generer_graphique_relations(chemin_csv):
    """Affiche la répartition des classes par position (T1, T2...) à partir du CSV."""
    df = pd.read_csv(chemin_csv)
    df = df.rename(columns={'target_label': 'Cible', 'relation': 'Relation'})
    
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x='Cible', hue='Relation', palette='Set2')
    
    plt.title("Distribution des Annotations par Position", fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("distribution_relations.png")
    plt.show()

# Dans c:\Users\user\Desktop\stage\dataset\visualisation.py

def generer_rapport_visuel(csv_path, output_image="matrice_heatmap.png"):
    """Génère le graphique de distribution ET la matrice de corrélation."""
    # 1. Chargement et Nettoyage
    df = pd.read_csv(csv_path)
    
    # On renomme pour avoir des noms propres dans les graphiques
    # 'target_label' devient 'Cible', 'relation' devient 'Relation'
    df = df.rename(columns={
        'target_label': 'Cible', 
        'relation': 'Relation',
        'similarity_score': 'Score'
    })

    # --- PARTIE 2 : MATRICE DE CORRÉLATION (HEATMAP) ---
    # Découpage des scores en tranches
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']
    df['Tranche_Score'] = pd.cut(df['Score'], bins=bins, labels=labels, include_lowest=True)
    
    # Création de la matrice (Attention : on utilise 'Relation' avec Majuscule ici)
    matrice = pd.crosstab(df['Tranche_Score'], df['Relation'])
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrice, annot=True, fmt='d', cmap='YlGnBu')
    plt.title('Matrice de Corrélation : Scores vs Relations (AI4Debunk)', fontweight='bold')
    plt.xlabel('Relations Annotées')
    plt.ylabel('Tranches de Score de Similarité')
    
    plt.savefig(output_image)
    print(f" Matrice sauvegardée sous : {output_image}")
    
    return matrice