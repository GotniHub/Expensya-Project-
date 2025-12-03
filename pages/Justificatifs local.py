import streamlit as st
import pandas as pd
import zipfile
import os
import shutil
import re
import requests
from io import BytesIO

# =============================
#        CONFIG DE BASE
# =============================
st.set_page_config(page_title="Gestion des Missions", layout="wide")
# si ta version de Streamlit supporte st.logo, garde la ligne ci-dessous
try:
    st.logo("LOGO.png", icon_image="Logom.png")
except Exception:
    pass
try:
    from streamlit_plotly_events import plotly_events
    PLOTLY_EVENTS_AVAILABLE = True
except Exception:
    PLOTLY_EVENTS_AVAILABLE = False

st.title("📂 Générateur et accès aux dossiers missions")

# =============================
#    VÉRIF AUTH & MISSIONS
# =============================
if "auth_user" not in st.session_state or st.session_state["auth_user"] is None:
    st.warning("⚠️ Aucun utilisateur connecté. Veuillez d’abord vous connecter depuis la page **Home**.")
    st.stop()

user = st.session_state["auth_user"]
missions = st.session_state.get("missions", [])
# =============================
#    SIDEBAR – Session Expensya
# =============================

session = st.session_state.get("current_period", "Aucune session")


# Choix mission
if len(missions) == 1:
    missions_selected = missions[:]  # liste avec une seule mission
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

#    SIDEBAR INFO USER + SESSION
# Bloc utilisateur (carte latérale)
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
    <p style="margin: 0; font-size: 0.9rem; opacity: 0.95;">🟢  Connecté avec succès</p>
</div>
"""
st.sidebar.markdown(card_html, unsafe_allow_html=True)

st.sidebar.markdown(f"""
<div style="
    padding: 12px;
    background-color: rgba(194, 221, 255, 0.55);
    border-left: 4px solid #2E86C1;
    border-radius: 6px;
    margin-bottom: 15px;
    font-size: 14px;
">
<b>🟢 Session Expensya</b><br>
{session}
</div>
""", unsafe_allow_html=True)

if st.sidebar.button("⏻ Déconnexion"):
    st.session_state["auth_user"] = None
    st.switch_page("home.py")

# =============================
#     LIEN ONEDRIVE (EXEMPLE)
# =============================
# Exctraction S1 2025 (actuel) :
# ONEDRIVE_URL = "https://adventplus-my.sharepoint.com/:u:/g/personal/igotni_adv-sud_fr/Ef8LL-Y_mNhOlCQlKHlQs1wBXzoorlA-dVNmoZ07zj3oNw?download=1"
# Exctraction S2 2024 :
# ONEDRIVE_URL = "https://adventplus-my.sharepoint.com/:u:/g/personal/igotni_adv-sud_fr/EahoQ8gXXhJLpKJy4FtfyvsBsKc7r60cII0KbVjkorzH6g?download=1"
# Extraction TR S1 2025 :

# ONEDRIVE_URL = "https://adventplus-my.sharepoint.com/:u:/g/personal/igotni_adv-sud_fr/EVAEu6MEKhVOqn3UhLlYSyEBNOF9OuzIaUxNd0zjqFLqaw?download=1"
# -------------------------
# Lien OneDrive défini lors de la connexion (en fonction de la période)
ONEDRIVE_URL = st.session_state.get("onedrive_url")

if not ONEDRIVE_URL:
    st.error("Aucun lien OneDrive n'est défini. Retourne à la page d'accueil et reconnecte-toi.")
    st.stop()

# =============================
#         UTILITAIRES
# =============================
def nettoyer_nom(nom: str) -> str:
    """Nettoie les noms de dossiers/fichiers pour compatibilité cross-OS."""
    return re.sub(r'[<>:"/\\|?*]', "_", str(nom).strip()).lower()

def lire_matrice(path: str) -> pd.DataFrame:
    """Essaye plusieurs noms d’onglets possibles pour la matrice."""
    xl = pd.ExcelFile(path)
    for sheet in xl.sheet_names:
        if "matrice" in sheet.lower():
            return xl.parse(sheet)
    # fallback : premier onglet
    return xl.parse(xl.sheet_names[0])

def lire_rapport(path: str) -> pd.DataFrame:
    """Lit l’onglet Rapport (ou tente un fallback)."""
    xl = pd.ExcelFile(path)
    if "Rapport" in xl.sheet_names:
        return xl.parse("Rapport")
    return xl.parse(xl.sheet_names[0])
from collections import defaultdict
def build_receipts_index_from_zipfile(zip_path: str) -> dict[str, list[tuple[str, bytes]]]:
    idx = defaultdict(list)
    with zipfile.ZipFile(zip_path, "r") as z:
        for name in z.namelist():
            if name.endswith("/"):      # ignorer les dossiers
                continue
            base = os.path.basename(name)
            if not base:
                continue
            m = re.match(r"^(\d+)", base)   # réf = chiffres au début du nom
            if not m:
                continue
            ref = m.group(1)
            try:
                data = z.read(name)        # bytes du fichier
            except Exception:
                continue
            idx[ref].append((base, data))
    return idx
# =============================
#        TRAITEMENT ZIP
# =============================
st.divider()
if st.button("🚀 Lancer le traitement"):
    try:
        st.info("⏳ Téléchargement du ZIP depuis la base…")
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Étape 1 : téléchargement
        response = requests.get(ONEDRIVE_URL, stream=True, timeout=300)
        if response.status_code != 200:
            st.error("❌ Erreur lors du téléchargement OneDrive")
            st.stop()

        total_size = int(response.headers.get('content-length', 0)) or None
        downloaded_size = 0
        zip_content = BytesIO()

        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                zip_content.write(chunk)
                if total_size:
                    downloaded_size += len(chunk)
                    progress = int(downloaded_size / total_size * 100)
                    progress_bar.progress(min(progress, 100))
                    status_text.text(f"Téléchargement... {progress}%")

        progress_bar.progress(100)
        status_text.text("✅ Téléchargement terminé. Traitement en cours...")

        # Remettre le curseur au début
        zip_content.seek(0)

        # Dossier temporaire propre
        temp_dir = "temp_result"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

        # Sauvegarde du ZIP principal
        outer_zip_path = os.path.join(temp_dir, "expensya_docs.zip")
        with open(outer_zip_path, "wb") as f:
            f.write(zip_content.getvalue())

        # Extraction du ZIP principal
        with zipfile.ZipFile(outer_zip_path, "r") as outer_zip:
            outer_zip.extractall(temp_dir)

        # Identifier les fichiers (rapport, matrice, zip interne)
        rapport_file, mapping_file, inner_zip_path = None, None, None
        xlsx_files, zip_files = [], []

        for file in os.listdir(temp_dir):
            p = os.path.join(temp_dir, file)
            if os.path.isdir(p):
                continue
            if file.lower().endswith(".xlsx"):
                xlsx_files.append(p)
            elif file.lower().endswith(".zip"):
                zip_files.append(p)

        # Détection rapport / matrice
        for p in xlsx_files:
            fname = os.path.basename(p).lower()
            if "matrice" in fname:
                mapping_file = p
            else:
                rapport_file = p

        # Sélection du ZIP interne (justificatifs)
        zip_candidates = [p for p in zip_files if os.path.abspath(p) != os.path.abspath(outer_zip_path)]
        if zip_candidates:
            inner_zip_path = max(zip_candidates, key=os.path.getsize)

        # Vérifications
        if not rapport_file or not mapping_file or not inner_zip_path:
            st.error("❌ Impossible de trouver Rapport, Matrice ou le ZIP interne (justificatifs).")
            st.write("DEBUG – xlsx:", [os.path.basename(p) for p in xlsx_files])
            st.write("DEBUG – zip:", [os.path.basename(p) for p in zip_files])
            st.stop()
        # Build receipts index directly from the inner ZIP (handles subfolders)
        st.session_state["receipts_index"] = build_receipts_index_from_zipfile(inner_zip_path)
        st.success(f"🔍 Index justificatifs construit pour {len(st.session_state['receipts_index'])} références.")
        # Lecture des données
        df = lire_rapport(rapport_file)
        df_map = lire_matrice(mapping_file)

        # Colonnes minimales
        needed = {"Client (Référence)", "Utilisateur", "Référence"}
        if not needed.issubset(df.columns):
            st.error(f"Colonnes manquantes dans le Rapport : {sorted(list(needed - set(df.columns)))}")
            st.stop()

        # Merge mapping (si dispo)
        if {"Client (Référence)", "Modification Code Expensya"}.issubset(df_map.columns):
            df = df.merge(
                df_map[["Client (Référence)", "Modification Code Expensya"]],
                on="Client (Référence)",
                how="left"
            )

        # Retirer un éventuel "Grand Totale" final
        if not df.empty:
            last_row = df.tail(1).astype(str).apply(lambda x: x.str.contains("Grand Totale", case=False).any(), axis=1).iloc[0]
            if last_row:
                df = df.iloc[:-1]

        # Mission finale / nettoyée
        df["Mission_Final"] = df.apply(
            lambda row: row.get("Modification Code Expensya") if pd.notna(row.get("Modification Code Expensya")) else row.get("Client (Référence)"),
            axis=1
        ).fillna("").replace("", "vide")
        df["Mission_Clean"] = df["Mission_Final"].apply(nettoyer_nom)
        
        # --- ⬇️ Sauvegarde pour le calendrier (réutilisable sans upload) ---
        # 1) Normaliser Date
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

        # 2) Construire MissionLib (si pas déjà présent)
        if "Modification Code Expensya" in df.columns:
            _map = dict(zip(
                df["Client (Référence)"].astype(str),
                df["Modification Code Expensya"].astype(str)
            ))
        else:
            _map = {}

        df["MissionLib"] = df["Client (Référence)"].astype(str).map(_map).fillna(df["Client (Référence)"].astype(str))

        # 3) Colonnes utiles pour le calendrier (léger)
        _cal_cols = ["Date", "Nom de la dépense", "Catégorie", "Utilisateur",
                    "Client (Référence)", "Référence", "MissionLib", "TTC (EUR)"]
        cal_base = df[_cal_cols].copy()

        # 4) Stocker en session
        st.session_state["cal_from_onedrive"] = cal_base
        st.session_state["mat_map"] = _map              # (optionnel) pour réappliquer MissionLib
        # Les justificatifs sont déjà mémorisés plus haut :
        # st.session_state["receipts_zip_path"], st.session_state["receipts_index"]

        # Extraire justificatifs vers un dossier
        justificatifs_dir = os.path.join(temp_dir, "justifs")
        os.makedirs(justificatifs_dir, exist_ok=True)
        with zipfile.ZipFile(inner_zip_path, "r") as zf:
            zf.extractall(justificatifs_dir)
        import os, re, zipfile
        # ... tu as déjà: justificatifs_dir rempli à partir de inner_zip_path

        def _norm_ref(s: str) -> str:
            # normalise une référence : garde les chiffres uniquement (ex: "Ref-009024.pdf" -> "9024")
            return re.sub(r"\D", "", str(s)).lstrip("0") or "0"

        # Construit l'index global une fois pour toutes
        receipts_index = {}

        with zipfile.ZipFile(inner_zip_path, "r") as zread:
            for name in zread.namelist():
                if name.endswith("/"):
                    continue
                base = os.path.basename(name)
                if not base:
                    continue
                # 1) si le fichier commence par la ref -> simple
                m = re.match(r"^(\d+)[\s_\-\.].*", base)
                if m:
                    ref_key = _norm_ref(m.group(1))
                    receipts_index.setdefault(ref_key, []).append(name)
                    continue
                # 2) sinon, on tente de trouver un bloc de chiffres au début ou après un séparateur
                m2 = re.search(r"(\d{3,})", base)
                if m2:
                    ref_key = _norm_ref(m2.group(1))
                    receipts_index.setdefault(ref_key, []).append(name)

        # Sauvegarde en session pour la page calendrier
        st.session_state["receipts_zip_path"] = inner_zip_path
        st.session_state["receipts_index"] = receipts_index

        # Filtrer le DF par missions sélectionnées
        # (on compare sur la valeur brute "Client (Référence)" ET sur "Modification Code Expensya")
        missions_lower = set(m.lower() for m in missions_selected)
        df_filt = df[
            df["Client (Référence)"].astype(str).str.lower().isin(missions_lower) |
            df["Mission_Final"].astype(str).str.lower().isin(missions_lower)
        ].copy()
        # Nombre de justificatifs pour les missions sélectionnées
        nb_justifs_missions = df_filt["Référence"].astype(str).nunique()
        st.info(f"📄 Justificatifs pour les missions sélectionnées : {nb_justifs_missions}")

        if df_filt.empty:
            st.warning("Aucune ligne du rapport ne correspond aux missions sélectionnées.")
            st.stop()

        # --- Création des dossiers missions ---
        grouped = df_filt.groupby("Mission_Final")

        for mission, group in grouped:
            mission_clean = nettoyer_nom(mission)
            mission_path = os.path.join(temp_dir, mission_clean)
            os.makedirs(mission_path, exist_ok=True)

            # Sauvegarde du rapport Excel par mission
            excel_path = os.path.join(mission_path, f"{mission_clean}.xlsx")
            group.to_excel(excel_path, index=False)

            # Répartition par (mois -> user)
            for _, row in group.iterrows():
                ref = str(row.get("Référence", "")).strip()
                user_name = nettoyer_nom(row.get("Utilisateur", "inconnu"))

                # Dossier mois (selon "Date" si dispo)
                date_val = pd.to_datetime(row.get("Date", pd.NaT), errors="coerce")
                mois_str = date_val.strftime("%B %Y") if pd.notna(date_val) else "inconnu"

                mois_dir = os.path.join(mission_path, mois_str)
                user_dir = os.path.join(mois_dir, user_name)
                os.makedirs(user_dir, exist_ok=True)

                # Copier les justificatifs correspondants (par ref)
                # On cherche les fichiers du ZIP interne qui commencent par la référence
                # 🔎 Recherche robuste des pièces (parcours récursif + normalisation)
                raw_ref = str(row.get("Référence", "")).strip()
                norm_ref = re.sub(r"\D", "", raw_ref).lstrip("0") or "0"

                for root, _, files in os.walk(justificatifs_dir):
                    for file in files:
                        base = os.path.basename(file)
                        m = re.match(r"^(\d+)[\s_\-\.]", base) or re.search(r"(\d{3,})", base)
                        if not m:
                            continue
                        key = re.sub(r"\D", "", m.group(1)).lstrip("0") or "0"
                        if key != norm_ref:
                            continue

                        nom_depense = str(row.get("Nom de la dépense", "inconnu")).strip()
                        categorie = str(row.get("Catégorie", "inconnu")).strip()
                        date_str = date_val.strftime("%Y-%m-%d") if pd.notna(date_val) else "inconnu"

                        _, ext = os.path.splitext(base)
                        new_name = f"{raw_ref}_{nom_depense}_{categorie}_{date_str}{ext}"
                        new_name = re.sub(r'[<>:"/\\|?*]', '_', new_name)

                        src = os.path.join(root, file)
                        dst = os.path.join(user_dir, new_name)
                        shutil.copy(src, dst)

        # Zippage de sortie pour téléchargement
        output = BytesIO()
        added_files = 0
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
            for mission in set(df_filt["Mission_Final"]):
                mission_clean = nettoyer_nom(mission)
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
            st.download_button("📥 Télécharger toutes vos missions", output, file_name=f"{user}_missions.zip")
        else:
            st.warning("⚠️ Aucun dossier généré.")
            st.write("Missions sélectionnées:", missions_selected)
            st.write("Dossiers générés:", os.listdir(temp_dir))

    except Exception as e:
        st.error(f"❌ Erreur : {e}")

# =============================
#     📅 CALENDRIER DES JUSTIFICATIFS
# =============================
# =============================
#     📅 CALENDRIER DES JUSTIFICATIFS
# =============================
st.divider()
st.subheader("📅 Calendrier des justificatifs (par date)")

st.markdown(
    "- Charge le **Rapport Expensya (.xlsx)** (onglet 'Rapport').  \n"
    "- Un **point = une dépense** colorée par **Catégorie**.  \n"
    "- Clique sur un point pour voir les justificatifs du jour."
)

# 1) Rapport : fichier importé (prioritaire) OU données OneDrive déjà traitées
cal_file = st.file_uploader(
    "Importer le fichier Rapport (.xlsx)", 
    type=["xlsx"], 
    key="cal_uploader"
)

if cal_file is not None:
    # Cas A : fichier importé
    cal_df = pd.read_excel(cal_file, sheet_name="Rapport")
    cal_df["Date"] = pd.to_datetime(cal_df["Date"], errors="coerce")

    # MissionLib depuis le mapping global si dispo (mat_map créée au moment du traitement OneDrive)
    map_dict = st.session_state.get("mat_map", {})
    if "MissionLib" not in cal_df.columns or cal_df["MissionLib"].isna().all():
        if map_dict:
            cal_df["MissionLib"] = (
                cal_df["Client (Référence)"]
                .astype(str)
                .map(map_dict)
                .fillna(cal_df["Client (Référence)"].astype(str))
            )
        else:
            cal_df["MissionLib"] = cal_df["Client (Référence)"].astype(str)
else:
    # Cas B : réutiliser les données du rapport déjà traitées depuis OneDrive
    cal_df = st.session_state.get("cal_from_onedrive")

# Garde-fou si rien n’est dispo
if cal_df is None or cal_df.empty:
    st.warning("Aucune donnée de calendrier disponible. Lance d'abord le traitement ou importe le Rapport.")
    st.stop()
# 🔹 Filtrer strictement sur les missions sélectionnées (y compris 106710)
missions_set = {m.strip().lower() for m in missions_selected}
cal_df = cal_df[
    cal_df["Client (Référence)"].astype(str).str.lower().isin(missions_set)
    | cal_df["MissionLib"].astype(str).str.lower().isin(missions_set)
]
# 2) Filtrer par missions sélectionnées (IMPORTANT)
missions_set = {m.strip().lower() for m in missions_selected}

cal_df = cal_df[
    cal_df["Client (Référence)"].astype(str).str.lower().isin(missions_set)
    | cal_df["MissionLib"].astype(str).str.lower().isin(missions_set)
]


# 3) Nettoyage de base
cal_df["Date"] = pd.to_datetime(cal_df["Date"], errors="coerce")
cal_df = cal_df.dropna(subset=["Date"])

# 4) Exclure les missions NO REFACT (toutes variantes)
cal_df["MissionLib"] = cal_df["MissionLib"].astype(str)
cal_df = cal_df[~cal_df["MissionLib"].str.contains("NO REFACT", case=False, na=False)]

# Garde-fou après filtre
if cal_df.empty:
    st.warning("Après filtres (missions + NO REFACT), aucune dépense trouvée.")
    st.stop()

# 5) Pastille d’état UX
if cal_file is None:
    st.caption("🟢 Données calendrier chargées depuis Base données (session courante).")
else:
    st.caption("📄 Données calendrier chargées depuis le fichier importé.")

# # 6) Matrice optionnelle (pour ajuster le libellé mission affiché, sans casser les filtres)
# mat_file = st.file_uploader(
#     "➕ (optionnel) Matrice Expensya (.xlsx) pour afficher le libellé mission",
#     type=["xlsx"],
#     key="mat_for_cal"
# )
# client_to_label = {}
# if mat_file is not None:
#     try:
#         _mat = pd.read_excel(mat_file)
#         if {"Client (Référence)", "Modification Code Expensya"}.issubset(_mat.columns):
#             client_to_label = dict(
#                 zip(
#                     _mat["Client (Référence)"].astype(str),
#                     _mat["Modification Code Expensya"].astype(str),
#                 )
#             )
#     except Exception:
#         pass

# if client_to_label:
#     cal_df["MissionLib"] = (
#         cal_df["Client (Référence)"]
#         .astype(str)
#         .map(client_to_label)
#         .fillna(cal_df["MissionLib"])
#     )

# # 7) ZIP justificatifs optionnel (pour preview / download)
# zip_for_calendar = st.file_uploader(
#     "➕ (optionnel) ZIP des justificatifs (export Expensya) — pour prévisualiser/télécharger",
#     type=["zip"],
#     key="zip_for_calendar"
# )


# # ====== DEBUG : comprendre pourquoi 228 ≠ 216 ======
# st.write("🚧 DEBUG — Lignes calendrier après filtres :", len(cal_df))

# # Nombre de lignes par mission
# st.write("Lignes par mission (Client (Référence)) :")
# st.dataframe(
#     cal_df.groupby("Client (Référence)")["Référence"]
#           .nunique()
#           .reset_index(name="Nb_lignes")
# )

# # Lignes suspectes : nom contenant 'total' ou montant vide
# suspect = cal_df[
#     cal_df["Nom de la dépense"].astype(str).str.contains("total", case=False, na=False)
#     | cal_df["TTC (EUR)"].isna()
# ]
# st.write("Lignes suspectes (TOTAL / montant NaN) :")
# st.dataframe(
#     suspect[["Référence", "Date", "Nom de la dépense", "Client (Référence)", "MissionLib", "TTC (EUR)"]]
# )

# (optionnel) export des lignes pour comparaison dans Excel
# suspect.to_excel("debug_suspect.xlsx", index=False)
# st.download_button("📥 Télécharger les lignes suspectes", open("debug_suspect.xlsx","rb"), "debug_suspect.xlsx")

# 8) Métriques (tu choisis ce que “Dépenses” représente)
nb_lignes = len(cal_df)
nb_refs_uniques = cal_df["Référence"].astype(str).nunique()

col1, col2, col3 = st.columns(3)
# 👉 Si tu veux le nombre de *lignes* :
# col1.metric("Dépenses", f"{nb_lignes:,}".replace(",", " "))

# 👉 Si tu préfères le nombre de références uniques :
col1.metric("Dépenses", f"{nb_refs_uniques:,}".replace(",", " "))

col2.metric("Utilisateurs uniques", cal_df["Utilisateur"].nunique())
col3.metric("Jours distincts", cal_df["Date"].dt.date.nunique())


# Filtres
with st.expander("🎛️ Filtres"):
    users = sorted(cal_df["Utilisateur"].dropna().unique().tolist())
    cats = sorted(cal_df["Catégorie"].dropna().unique().tolist())
    sel_users = st.multiselect("Utilisateurs", users, default=users)
    sel_cats = st.multiselect("Catégories", cats, default=cats)
    date_min = cal_df["Date"].min().date()
    date_max = cal_df["Date"].max().date()
    date_range = st.date_input("Période", (date_min, date_max))

cal_f = cal_df[
    cal_df["Utilisateur"].isin(sel_users) &
    cal_df["Catégorie"].isin(sel_cats) &
    (cal_df["Date"].dt.date >= pd.to_datetime(date_range[0]).date()) &
    (cal_df["Date"].dt.date <= pd.to_datetime(date_range[-1]).date())
].copy()

if cal_f.empty:
    st.info("Aucune dépense ne correspond aux filtres actuels.")
    st.stop()

# ---------- Vue timeline cliquable (1 barre par utilisateur & jour) ----------

# 1) Préparer un enregistrement par (Utilisateur, Jour)
ev = cal_f.copy()
ev["Début"] = ev["Date"].dt.floor("D")
ev["Fin"]   = ev["Début"] + pd.Timedelta(days=1)

# agrégation par jour & utilisateur
agg = (
    ev.groupby(["Utilisateur", "Début"])
    .agg(
        CatList = ("Catégorie", lambda s: sorted(set(s))),
        MissionLib = ("MissionLib", "first"),
        Refs = ("Référence", lambda s: list(s))
    )
    .reset_index()
)
# Catégorie2 = "Mixte" si plusieurs catégories le même jour, sinon la seule catégorie
agg["Catégorie2"] = agg["CatList"].apply(lambda lst: "Mixte" if len(lst) > 1 else lst[0])
agg["Fin"] = agg["Début"] + pd.Timedelta(days=1)
agg["Date_str"] = agg["Début"].dt.strftime("%Y-%m-%d")  # pour le clic

# --- Préparer texte lisible pour la bulle (hover) ---
# Convertit la liste de références en une chaîne avec des puces
def join_refs(lst):
    if not lst:
        return ""
    return "<br>• ".join(str(x) for x in lst)

# Si tu veux aussi montrer les montants par jour, construis Montants_join (optionnel)
if "TTC (EUR)" in cal_f.columns:
    ref_to_amount = cal_f.dropna(subset=["Référence"]).drop_duplicates("Référence").set_index("Référence")["TTC (EUR)"].to_dict()
else:
    ref_to_amount = {}

agg["Refs_join"] = agg["Refs"].apply(lambda lst: join_refs(lst))

def join_amounts(lst):
    out = []
    for r in lst:
        amt = ref_to_amount.get(str(r), None)
        if amt is not None and pd.notna(amt):
            try:
                out.append(f"{r} : {float(amt):.2f}€")
            except Exception:
                out.append(f"{r} : {amt}")
        else:
            out.append(str(r))
    return ", ".join(out)

if ref_to_amount:
    agg["Montants_join"] = agg["Refs"].apply(lambda lst: join_amounts(lst))
else:
    agg["Montants_join"] = ""

# ============================================
# Timeline avec bulle lisible (Catégorie — Nom (+ montant))
# ============================================

# 1) Agrégation par (Utilisateur, Jour) avec une liste "Items" lisible
def _items_for_day(dfday):
    out = []
    for _, r in dfday.iterrows():
        cat = str(r.get("Catégorie", "")).strip()
        nom = str(r.get("Nom de la dépense", "")).strip()
        amt = r.get("TTC (EUR)", None)
        if amt is not None and pd.notna(amt):
            try:
                line = f"{cat} — {nom} ({float(amt):.2f}€)"
            except Exception:
                line = f"{cat} — {nom} ({amt})"
        else:
            line = f"{cat} — {nom}"
        out.append(line)
    return out

# Pour fabriquer "Items", on regroupe ev et on reprend les mêmes lignes du groupe
def _items_grouped(g):
    # g est un Series de noms de dépense, on récupère l'index dans ev
    return _items_for_day(ev.loc[g.index])

agg = (
    ev.groupby(["Utilisateur", "Début"])
    .agg(
        CatList=("Catégorie", lambda s: sorted(set(s))),
        MissionLib=("MissionLib", "first"),
        Refs=("Référence", lambda s: list(s)),                    # tu peux garder si utile ailleurs
        Items=("Nom de la dépense", _items_grouped)               # 👈 liste lisible
    )
    .reset_index()
)

# Catégorie "Mixte" si plusieurs catégories le même jour
# Compter le nombre de notes par (Utilisateur, Jour)
agg["NoteCount"] = agg["Items"].apply(lambda L: len(L) if isinstance(L, list) else 0)

# Si +1 note le même jour => "Mixte" (même si toutes les notes ont la même catégorie)
# Sinon, on affiche la catégorie unique s'il y en a une
def _cat_mixed(row):
    if row["NoteCount"] > 1:
        return "Mixte"
    lst = row["CatList"]
    return lst[0] if isinstance(lst, list) and len(lst) >= 1 else "Inconnue"

agg["Catégorie2"] = agg.apply(_cat_mixed, axis=1)

agg["Fin"] = agg["Début"] + pd.Timedelta(days=1)
agg["Date_str"] = agg["Début"].dt.strftime("%Y-%m-%d")

# Liste à puces pour la bulle
def join_items(lst):
    if not lst:
        return ""
    return "• " + "<br>• ".join(str(x) for x in lst)
agg["Items_join"] = agg["Items"].apply(join_items)

# 2) Tracer la timeline avec custom_data pour le hover
import plotly.express as px
fig = px.timeline(
    agg,
    x_start="Début",
    x_end="Fin",
    y="Utilisateur",
    color="Catégorie2",
    hover_name="Utilisateur",
    # customdata indices:
    # 0 Utilisateur, 1 Date_str, 2 Items_join, 3 MissionLib, 4 Catégorie2
    custom_data=["Utilisateur", "Date_str", "Items_join", "MissionLib", "Catégorie2"],
    color_discrete_sequence=px.colors.qualitative.Set2
)
fig.update_yaxes(autorange="reversed")

# Hachurer les jours Mixte
for tr in fig.data:
    if tr.name == "Mixte":
        tr.marker.pattern.shape = "/"
        tr.marker.pattern.fillmode = "overlay"
        tr.marker.line.width = 0.6

# 3) Bulle personnalisée
hover_template = (
    "<b>%{customdata[0]}</b> — %{customdata[1]}<br>"
    "Mission : %{customdata[3]}<br><br>"
    "%{customdata[2]}"                    # liste à puces (Items_join)
    "<extra></extra>"
)
fig.update_traces(hovertemplate=hover_template)

# 4) Mise en forme & affichage
fig.update_layout(
    height=520,
    margin=dict(l=20, r=20, t=30, b=20),
    xaxis_title="Date",
    yaxis_title="Utilisateur",
    legend_title_text="Catégorie (Mixte = plusieurs catégories le même jour)",
    uirevision="cal_v1"
)
st.plotly_chart(fig, use_container_width=True)




# Index justificatifs si ZIP fourni
# --------- Accès aux justificatifs ---------
ref_files = {}
zf = None

# # 1) Cas A : l'utilisateur a uploadé un ZIP -> priorité
# if zip_for_calendar is not None:
#     try:
#         zf = zipfile.ZipFile(zip_for_calendar)
#         # (re)construit l'index à partir de l'upload
#         ref_files = {}
#         for name in zf.namelist():
#             base = os.path.basename(name)
#             if not base:
#                 continue
#             # même normalisation que plus haut
#             m = re.match(r"^(\d+)[\s_\-\.].*", base) or re.search(r"(\d{3,})", base)
#             if m:
#                 key = re.sub(r"\D","", m.group(1)).lstrip("0") or "0"
#                 ref_files.setdefault(key, []).append(name)
#     except Exception as e:
#         st.warning(f"Impossible de lire le ZIP justificatifs uploadé : {e}")
#         zf = None

# 2) Cas B : aucun upload -> on réutilise l'index et le ZIP internes OneDrive
if zf is None and "receipts_zip_path" in st.session_state and "receipts_index" in st.session_state:
    try:
        internal_zip = st.session_state["receipts_zip_path"]
        zf = zipfile.ZipFile(internal_zip)
        ref_files = st.session_state.get("receipts_index", {})
    except Exception as e:
        st.warning(f"Impossible d'ouvrir le ZIP interne OneDrive : {e}")
        zf = None
        ref_files = {}

# =============================
# 📦 Exporter justificatifs par utilisateur / période
# (à placer après la construction de `ref_files` et `zf`, avant l'expander "Détails des jours...")
# =============================
st.markdown("### 📦 Exporter les justificatifs par utilisateur et période")

if zf is None:
    st.info("➡️ Pour activer l'export, fournis le **ZIP des justificatifs** (ou exécute le traitement OneDrive).")
else:
    # Sélection des utilisateurs (par défaut ceux visibles dans le calendrier filtré)
    users_export = sorted(cal_f["Utilisateur"].dropna().unique().tolist())
    pick_users = st.multiselect("Utilisateurs à inclure", users_export, default=users_export)

    # Sélection de la période (par défaut la même que les filtres actuels)
    date_min_exp = cal_f["Date"].min().date()
    date_max_exp = cal_f["Date"].max().date()
    dr_export = st.date_input("Période d'export", (date_min_exp, date_max_exp), key="export_period")

    # Filtrer les lignes à exporter
    df_export = cal_f[
        cal_f["Utilisateur"].isin(pick_users) &
        (cal_f["Date"].dt.date >= pd.to_datetime(dr_export[0]).date()) &
        (cal_f["Date"].dt.date <= pd.to_datetime(dr_export[-1]).date())
    ].copy()

    st.caption(f"🧾 {len(df_export)} notes de frais retenues pour l'export.")

    # Lancer l'export
    col_exp_btn, _ = st.columns([1,4])
    if col_exp_btn.button("📥 Générer le ZIP des justificatifs", key="btn_export_zip"):
        from io import BytesIO
        import re, os, zipfile

        out = BytesIO()
        added, missing = 0, 0

        with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for _, r in df_export.iterrows():
                raw_ref = str(r.get("Référence", "")).strip()
                norm_ref = re.sub(r"\D", "", raw_ref).lstrip("0") or "0"
                user     = re.sub(r'[<>:"/\\|?*]', "_", str(r.get("Utilisateur", "inconnu")).strip())
                date_day = r["Date"].strftime("%Y-%m-%d")
                mission  = str(r.get("MissionLib", r.get("Client (Référence)", "mission"))).strip()
                mission  = re.sub(r'[<>:"/\\|?*]', "_", mission)

                files = ref_files.get(norm_ref, [])

                if not files:
                    missing += 1
                    continue

                for path in files:
                    data = zf.read(path)
                    base = os.path.basename(path)

                    # renommage + arborescence ZIP : Utilisateur/Date/Mission/nomfichier
                    # (tu peux simplifier si tu veux)
                    arcname = f"{user}/{date_day}/{mission}/{base}"
                    zout.writestr(arcname, data)
                    added += 1

        if added == 0:
            st.warning("Aucun fichier n'a pu être ajouté (références introuvables dans le ZIP).")
        else:
            out.seek(0)
            st.success(f"✅ ZIP prêt : {added} fichier(s) ajouté(s)" + (f", {missing} référence(s) sans pièce" if missing else ""))
            st.download_button(
                "⬇️ Télécharger le ZIP export",
                out,
                file_name="justificatifs_export.zip",
                mime="application/zip",
                key="dl_export_zip"
            )
# =============================
# 🔎 Sélection manuelle (fallback) + Preview justificatifs
# =============================
st.markdown("### 🔎 Sélection manuelle (fallback)")

# 1) Choix utilisateur puis jour (selon cal_f déjà filtré)
users_opts = ["—"] + sorted(cal_f["Utilisateur"].dropna().unique().tolist())
user_pick = st.selectbox("Utilisateur", users_opts, index=0, key="fallback_user")

day_pick = "—"
if user_pick != "—":
    day_opts = ["—"] + sorted(
        cal_f.loc[cal_f["Utilisateur"] == user_pick, "Date"].dt.strftime("%Y-%m-%d").unique().tolist()
    )
    day_pick = st.selectbox("Jour", day_opts, index=0, key="fallback_day")

# 2) Détails + preview si on a un couple valide
if user_pick != "—" and day_pick != "—":
    st.subheader(f"🗂️ Détails pour {user_pick} — {day_pick}")

    clicked_df = cal_f[
        (cal_f["Utilisateur"] == user_pick) &
        (cal_f["Date"].dt.strftime('%Y-%m-%d') == day_pick)
    ].sort_values("Date")

    if clicked_df.empty:
        st.info("Aucune note de frais pour ce point.")
    else:
        import base64, os

        for i, r in clicked_df.iterrows():
            ref = str(r.get("Référence", "")).strip()
            mission_txt = r.get("MissionLib", r.get("Client (Référence)", ""))
            montant = ""
            if "TTC (EUR)" in r and pd.notna(r["TTC (EUR)"]):
                try:
                    montant = f" — {float(r['TTC (EUR)']):.2f} EUR"
                except Exception:
                    pass


            with st.expander(
                f"📄 {r['Catégorie']} · {r['Nom de la dépense']}{montant} · _Mission : {mission_txt}_",
                expanded=True if len(clicked_df) == 1 else False
            ):
                st.write(f"**Référence** : `{ref}`")
                if pd.notna(r["Date"]):
                    st.write(f"**Heure** : {r['Date'].strftime('%H:%M')}")

                if zf is None:
                    st.caption("🛈 Fournis le ZIP des justificatifs ci-dessus pour prévisualiser et télécharger les fichiers.")
                else:
                    # 1) Trouver les pièces liées à la référence
                    norm_ref = re.sub(r"\D", "", ref).lstrip("0") or "0"
                    files = ref_files.get(norm_ref, [])

                    if not files:
                        st.warning("Aucune pièce jointe trouvée pour cette référence.")
                    else:
                        # 2) <<< ICI LE SLIDER DE TAILLE >>>
                        prev_h = st.slider(
                            "Taille de prévisualisation (px)",
                            min_value=150,
                            max_value=900,
                            value=340,
                            step=10,
                            key=f"prev_h_{ref or i}"
                        )

                        # 3) Affichage des pièces : aperçu + Voir en grand + Télécharger
                        # --- juste avant la boucle des fichiers
                        import streamlit.components.v1 as components
                        import base64

                        for j, path in enumerate(files):
                            data = zf.read(path)
                            name = os.path.basename(path)
                            ext = name.lower().split(".")[-1]

                            # 1) placeholder plein-largeur pour l’affichage "grand"
                            row_key = f"{norm_ref}_{j}"
                            viewer_ph = st.empty()   # on y rendra le grand affichage si demandé

                            # 2) ligne de contrôles : Aperçu + Voir en grand + Télécharger
                            c_preview, c_view, c_dl = st.columns([6, 2, 2])

                            if ext in ("jpg", "jpeg", "png", "gif", "webp"):
                                # Aperçu image (contrôlé par le slider prev_h)
                                with c_preview:
                                    st.image(data, caption=name, width=prev_h)

                                # Bouton "Voir en grand"
                                with c_view:
                                    if st.button("👁️ Voir en grand", key=f"view_img_{row_key}"):
                                        st.session_state[f"open_{row_key}"] = True

                                # Téléchargement
                                with c_dl:
                                    st.download_button("⬇️ Télécharger", data, file_name=name, key=f"dl_img_{row_key}")

                                # Rendu plein écran (en dehors des colonnes)
                                if st.session_state.get(f"open_{row_key}"):
                                    with viewer_ph.container():
                                        st.markdown(f"### 🔎 {name}")
                                        st.image(data, use_container_width=True)
                                        if st.button("Fermer", key=f"close_img_{row_key}"):
                                            st.session_state[f"open_{row_key}"] = False
                                            viewer_ph.empty()

                            elif ext == "pdf":
                                b64 = base64.b64encode(data).decode("utf-8")

                                # Aperçu PDF (hauteur contrôlée par le slider)
                                with c_preview:
                                    components.html(
                                        f'<iframe src="data:application/pdf;base64,{b64}" '
                                        f'width="100%" height="{prev_h}px" style="border:0;"></iframe>',
                                        height=prev_h + 20,
                                        scrolling=True
                                    )

                                # Bouton "Voir en grand"
                                with c_view:
                                    if st.button("👁️ Voir en grand", key=f"view_pdf_{row_key}"):
                                        st.session_state[f"open_{row_key}"] = True

                                # Téléchargement
                                with c_dl:
                                    st.download_button("⬇️ Télécharger", data, file_name=name, key=f"dl_pdf_{row_key}")

                                # Rendu plein écran (en dehors des colonnes)
                                if st.session_state.get(f"open_{row_key}"):
                                    with viewer_ph.container():
                                        st.markdown(f"### 🔎 {name}")
                                        components.html(
                                            f'<iframe src="data:application/pdf;base64,{b64}" '
                                            f'width="100%" height="800px" style="border:0;"></iframe>',
                                            height=820,
                                            scrolling=True
                                        )
                                        if st.button("Fermer", key=f"close_pdf_{row_key}"):
                                            st.session_state[f"open_{row_key}"] = False
                                            viewer_ph.empty()

                            else:
                                with c_preview:
                                    st.caption(f"Aperçu indisponible pour « .{ext} ».")
                                with c_view:
                                    st.button("👁️ Voir en grand", key=f"view_other_{row_key}", disabled=True)
                                with c_dl:
                                    st.download_button("⬇️ Télécharger", data, file_name=name, key=f"dl_other_{row_key}")



# Jours multi-notes (optionnel)
with st.expander("🔎 Détails des jours avec plusieurs notes de frais"):
    tmp = cal_f.copy()
    tmp["Jour"] = tmp["Date"].dt.floor("D")
    g = tmp.groupby(["Utilisateur", "Jour"]).size().reset_index(name="count")
    multi = g[g["count"] > 1].sort_values(["Utilisateur", "Jour"])
    if multi.empty:
        st.write("Aucun jour ne contient plusieurs notes de frais.")
    else:
        for _, row in multi.iterrows():
            u, d = row["Utilisateur"], row["Jour"]
            sub = tmp[(tmp["Utilisateur"] == u) & (tmp["Jour"] == d)].sort_values("Date")
            st.markdown(
                f"**{u} — {d.strftime('%Y-%m-%d')}**  \n"
                f"_Catégories:_ {', '.join(sorted(set(sub['Catégorie'])))}  \n"
                f"_Nombre de notes:_ {len(sub)}"
            )
            for _, r in sub.iterrows():
                montant = ""
                if "TTC (EUR)" in r and pd.notna(r["TTC (EUR)"]):
                    montant = f" — {r['TTC (EUR)']:.2f} EUR"
                st.write(f"- **{r['Catégorie']}** · {r['Nom de la dépense']}{montant}")




# =============================
#      PIED DE PAGE
# =============================
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem; margin-top: 3rem; 
background: linear-gradient(to right, #f8f9fa, #e9ecef); border-radius: 10px;">
    <p style="font-size: 1.2rem; margin-bottom: 0.5rem;">
            <strong>ADVENT+ - Expensya Justificatifs Manager</strong>
    </p>
    <p style="margin-bottom: 0.5rem;">Internal Distribution Analysis & Automation Platform - v1.0</p>
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
</div>
""", unsafe_allow_html=True)
