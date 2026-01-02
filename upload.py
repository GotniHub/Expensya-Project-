import streamlit as st
import pandas as pd
import zipfile
import os
import shutil
import re
from io import BytesIO

st.title("📂 Générateur de dossiers missions (avec Matrice Expensya)")

st.write("Cette application lit un fichier Excel (onglet **Rapport**), un fichier de correspondance (**Matrice Expensya**) et un ZIP de justificatifs, puis génère automatiquement une arborescence par mission corrigée.")

# --- Upload des fichiers
excel_file = st.file_uploader("📑 Importer le fichier Excel (Rapport)", type=["xlsx"])
mapping_file = st.file_uploader("📑 Importer le fichier Matrice Expensya", type=["xlsx"])
zip_file = st.file_uploader("🗜️ Importer le dossier ZIP des pièces justificatives", type=["zip"])

def nettoyer_nom(nom, max_len=60):
    nom = re.sub(r'[<>:"/\\|?*]', "_", str(nom).strip())
    return nom[:max_len]


if excel_file and zip_file and mapping_file:
    if st.button("🚀 Lancer le traitement"):
        # Charger Excel principal (onglet Rapport)
        df = pd.read_excel(excel_file, sheet_name="Rapport")

        # Charger la Matrice Expensya
        df_map = pd.read_excel(mapping_file, sheet_name="Matrice Expensya")

        # Vérification colonnes
        colonnes_obligatoires = ["Référence", "Utilisateur", "Client (Référence)"]
        if not all(col in df.columns for col in colonnes_obligatoires):
            st.error(f"❌ Les colonnes {colonnes_obligatoires} doivent être présentes dans l'onglet 'Rapport'.")
        elif not all(col in df_map.columns for col in ["Client (Référence)", "Modification Code Expensya"]):
            st.error("❌ La Matrice Expensya doit contenir 'Client (Référence)' et 'Modification Code Expensya'.")
        else:
            # Fusion entre rapport et matrice expensya
            df = df.merge(
                df_map[["Client (Référence)", "Modification Code Expensya"]],
                on="Client (Référence)",
                how="left"
            )

            # Retirer la dernière ligne si elle contient "Grand Totale"
            if df.tail(1).astype(str).apply(lambda x: x.str.contains("Grand Totale", case=False).any(), axis=1).any():
                df = df.iloc[:-1]

            # Mission finale : si correspondance → Modification Code Expensya
            # sinon garder valeur initiale
            df["Mission_Final"] = df.apply(
                lambda row: row["Modification Code Expensya"]
                if pd.notna(row["Modification Code Expensya"])
                else row["Client (Référence)"],
                axis=1
            )

            # Si mission vide → dossier "VIDE"
            df["Mission_Final"] = df["Mission_Final"].fillna("").replace("", "VIDE")

            # Nettoyer pour usage dossier
            df["Mission_Clean"] = df["Mission_Final"].apply(lambda x: nettoyer_nom(x).lower())

            # Créer dossier temporaire
            import tempfile
            temp_dir = tempfile.mkdtemp(prefix="exp_")  # ex: C:\Users\...\AppData\Local\Temp\exp_xxxx

            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            os.makedirs(temp_dir)

            # Extraire ZIP justificatifs
            justificatifs_dir = os.path.join(temp_dir, "justifs")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(justificatifs_dir)

            # Groupement par mission corrigée
            grouped = df.groupby("Mission_Clean")

            for mission, group in grouped:
                mission_path = os.path.join(temp_dir, mission)
                os.makedirs(mission_path, exist_ok=True)

                # Sauvegarder Excel par mission
                excel_path = os.path.join(mission_path, f"{mission}.xlsx")
                group.to_excel(excel_path, index=False)

                # Créer sous-dossier justificatifs
                justif_path = os.path.join(mission_path, "justificatifs")
                os.makedirs(justif_path, exist_ok=True)

                # Associer justificatifs par référence
                for _, row in group.iterrows():
                    ref = str(row["Référence"]).strip()
                    user = nettoyer_nom(row["Utilisateur"])
                    user_dir = os.path.join(justif_path, user)
                    os.makedirs(user_dir, exist_ok=True)

                    # Recherche du fichier justificatif
                    for file in os.listdir(justificatifs_dir):
                        if file.startswith(ref):
                            shutil.copy(
                                os.path.join(justificatifs_dir, file),
                                os.path.join(user_dir, file)
                            )

            # Créer archive finale
            output = BytesIO()
            with zipfile.ZipFile(output, "w") as zipf:
                for root, _, files in os.walk(temp_dir):
                    for file in files:
                        full_path = os.path.join(root, file)
                        rel_path = os.path.relpath(full_path, temp_dir)
                        zipf.write(full_path, rel_path)

            output.seek(0)
            st.success("✅ Traitement terminé avec la Matrice Expensya !")
            st.download_button("📥 Télécharger le ZIP final", output, file_name="missions_resultat.zip")
