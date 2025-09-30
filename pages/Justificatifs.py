import streamlit as st
import pandas as pd
import zipfile
import os
import shutil
import re
import requests
from io import BytesIO

st.set_page_config(page_title="Gestion des Missions", layout="wide")
st.logo("LOGO.png", icon_image="Logom.png")
st.title("📂 Générateur et accès aux dossiers missions")

# Vérifier la connexion
if "auth_user" not in st.session_state or st.session_state["auth_user"] is None:
    st.warning("⚠️ Aucun utilisateur connecté. Veuillez d’abord vous connecter depuis la page **Home**.")
    st.stop()

# Si connecté
user = st.session_state["auth_user"]
missions = st.session_state.get("missions", [])
# Choix mission
if len(missions) == 1:
    missions_selected = missions  # liste avec une seule mission
    st.info(f"✅ Mission assignée automatiquement : {missions[0]}")
else:
    missions_selected = st.multiselect(
        "📌 Sélectionnez vos missions :", missions, default=missions
    )
    if missions_selected:
        st.success(f"Missions sélectionnées : {', '.join(missions_selected)}")
    else:
        st.warning("⚠️ Aucune mission sélectionnée, merci d'en choisir au moins une.")
        st.stop()


user = st.session_state.get("auth_user", "Invité")

# Bloc utilisateur avec design 3D Advent+
card_html = f"""
<div style="
    background: linear-gradient(135deg, rgba(30,45,80,0.75), rgba(46,134,193,0.75)); 
    color: white; 
    padding: 1.2rem; 
    border-radius: 12px;
    text-align: center;
    margin-bottom: 1.5rem;
    box-shadow: 0 6px 15px rgba(0,0,0,0.25);
    transform: perspective(1000px) rotateX(2deg) rotateY(-1deg);
">
    <h4 style="margin: 0; font-size: 1.2rem; font-weight: bold;">👤 {user}</h4>
    <p style="margin: 0; font-size: 0.9rem; opacity: 0.95;">✔️  Connecté avec succès</p>
</div>
"""

st.sidebar.markdown(card_html, unsafe_allow_html=True)


# Bouton déconnexion stylé
if st.sidebar.button("🚪 Déconnexion"):
    st.session_state["auth_user"] = None
    st.switch_page("home.py")

# -------------------------
# Lien OneDrive (exemple)
# -------------------------
ONEDRIVE_URL = "https://adventplus-my.sharepoint.com/:u:/g/personal/igotni_adv-sud_fr/EahoQ8gXXhJLpKJy4FtfyvsBsKc7r60cII0KbVjkorzH6g?download=1"

# -------------------------
# Traitement
# -------------------------
def nettoyer_nom(nom):
    return re.sub(r'[<>:"/\\|?*]', "_", str(nom).strip())

if st.button("🚀 Lancer le traitement"):
    try:
        st.info("⏳ Téléchargement du ZIP depuis Base de données...")

        # Créer une barre de progression
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Étape 1 : téléchargement
        response = requests.get(ONEDRIVE_URL, stream=True)
        if response.status_code != 200:
            st.error("❌ Erreur lors du téléchargement OneDrive")
            st.stop()

        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0
        zip_content = BytesIO()

        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                zip_content.write(chunk)
                downloaded_size += len(chunk)
                progress = int(downloaded_size / total_size * 100)
                progress_bar.progress(progress)
                status_text.text(f"Téléchargement... {progress}%")

        progress_bar.progress(100)
        status_text.text("✅ Téléchargement terminé. Traitement en cours...")

        # Continue ton traitement avec zip_content
        zip_content.seek(0)

        # Créer dossier temporaire
        temp_dir = "temp_result"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)

        # Sauvegarde du ZIP principal à partir de zip_content
        outer_zip_path = os.path.join(temp_dir, "expensya_docs.zip")
        with open(outer_zip_path, "wb") as f:
            f.write(zip_content.getvalue())


        # Extraction du ZIP principal
        with zipfile.ZipFile(outer_zip_path, "r") as outer_zip:
            outer_zip.extractall(temp_dir)

        # Identifier les fichiers (rapport, matrice, zip interne)
        rapport_file, mapping_file, inner_zip_path = None, None, None
        for file in os.listdir(temp_dir):
            if file.endswith(".xlsx") and "Matrice" in file:
                mapping_file = os.path.join(temp_dir, file)
            elif file.endswith(".xlsx"):
                rapport_file = os.path.join(temp_dir, file)
            elif file.endswith(".zip"):
                inner_zip_path = os.path.join(temp_dir, file)

        if not rapport_file or not mapping_file or not inner_zip_path:
            st.error("❌ Impossible de trouver Rapport, Matrice ou le ZIP interne")
            st.stop()

        # Extraire le ZIP interne (justificatifs)
        justificatifs_dir = os.path.join(temp_dir, "justifs")
        with zipfile.ZipFile(inner_zip_path, "r") as inner_zip:
            inner_zip.extractall(justificatifs_dir)

        # Charger fichiers Excel
        df = pd.read_excel(rapport_file, sheet_name="Rapport")
        df_map = pd.read_excel(mapping_file, sheet_name="Matrice Expensya")

        # Vérification colonnes
        if not all(col in df.columns for col in ["Référence", "Utilisateur", "Client (Référence)"]):
            st.error("❌ Colonnes manquantes dans Rapport")
            st.stop()
        if not all(col in df_map.columns for col in ["Client (Référence)", "Modification Code Expensya"]):
            st.error("❌ Colonnes manquantes dans Matrice Expensya")
            st.stop()

        # Fusion
        df = df.merge(
            df_map[["Client (Référence)", "Modification Code Expensya"]],
            on="Client (Référence)",
            how="left"
        )

        # Retirer "Grand Totale"
        if df.tail(1).astype(str).apply(lambda x: x.str.contains("Grand Totale", case=False).any(), axis=1).any():
            df = df.iloc[:-1]

        # Mission finale
        df["Mission_Final"] = df.apply(
            lambda row: row["Modification Code Expensya"] if pd.notna(row["Modification Code Expensya"]) else row["Client (Référence)"],
            axis=1
        ).fillna("").replace("", "vide")
        df["Mission_Clean"] = df["Mission_Final"].apply(lambda x: nettoyer_nom(x).lower())

        # Split
        grouped = df.groupby("Mission_Final")
        for mission, group in grouped:
            mission_clean = nettoyer_nom(mission).lower()
            mission_path = os.path.join(temp_dir, mission_clean)
            os.makedirs(mission_path, exist_ok=True)

            excel_path = os.path.join(mission_path, f"{mission}.xlsx")
            group.to_excel(excel_path, index=False)

            justif_path = os.path.join(mission_path, "justificatifs")
            os.makedirs(justif_path, exist_ok=True)

            for _, row in group.iterrows():
                ref = str(row["Référence"]).strip()
                user_name = nettoyer_nom(row["Utilisateur"])
                user_dir = os.path.join(justif_path, user_name)
                os.makedirs(user_dir, exist_ok=True)

                for file in os.listdir(justificatifs_dir):
                    if file.startswith(ref):
                        shutil.copy(
                            os.path.join(justificatifs_dir, file),
                            os.path.join(user_dir, file)
                        )

        # Archive uniquement les missions sélectionnées
        output = BytesIO()
        added_files = 0

        with zipfile.ZipFile(output, "w") as zipf:
            for mission in missions_selected:
                mission_clean = nettoyer_nom(mission).lower()
                mission_dir = os.path.join(temp_dir, mission_clean)
                if os.path.exists(mission_dir):
                    for root, _, files in os.walk(mission_dir):
                        for file in files:
                            full_path = os.path.join(root, file)
                            rel_path = os.path.relpath(full_path, temp_dir)
                            zipf.write(full_path, rel_path)
                            added_files += 1


        if added_files > 0:
            output.seek(0)
            st.success("✅ Traitement terminé, toutes vos missions sont prêtes.")
            st.download_button(
                "📥 Télécharger toutes vos missions",
                output,
                file_name=f"{user}_missions.zip"
            )
        else:
            st.warning("⚠️ Aucun dossier trouvé pour votre compte.")
            st.write("Missions sélectionnées:", missions_selected)
            st.write("Dossiers générés:", os.listdir(temp_dir))




    except Exception as e:
        st.error(f"❌ Erreur : {e}")

st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem; margin-top: 3rem; background: linear-gradient(to right, #f8f9fa, #e9ecef); border-radius: 10px;">
    <p style="font-size: 1.2rem; margin-bottom: 0.5rem;">
            <strong>ADVENT+ - Expensya Justificatifs Manager</strong>
    </p>
    <p style="margin-bottom: 0.5rem;"> Internal Distribution Analysis & Automation Platform - v1.0</p>
    <p style="font-size: 0.9rem; margin-top: 0.8rem;">
        🔹 Génération automatique de dossiers missions • <br>
        🔹 Gestion sécurisée des justificatifs clients • <br>
        🔹 Intégration OneDrive & Expensya • <br>
        🔹 Contrôle utilisateur par authentification
    </p>
    <p style="font-size: 0.8rem; margin-top: 1rem;">
        <strong>🔒 Confidentialité :</strong> Usage interne réservé à <b>ADVENT+</b> • 
        Accès restreint par login/mot de passe
    </p>
    <p style="font-size: 0.8rem; margin-top: 1rem;">
        <a href="#" style="color: #2E86C1; text-decoration:none;">📘 Documentation</a> |
        <a href="#" style="color: #2E86C1; text-decoration:none;">🔐 Politique de confidentialité</a>
    </p>
</div>
""", unsafe_allow_html=True)