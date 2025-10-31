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
# Init session keys
if "receipts_index" not in st.session_state:
    st.session_state["receipts_index"] = {}

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
# Exctraction S2 2024 :
# ONEDRIVE_URL = "https://adventplus-my.sharepoint.com/:u:/g/personal/igotni_adv-sud_fr/EahoQ8gXXhJLpKJy4FtfyvsBsKc7r60cII0KbVjkorzH6g?download=1"
# -------------------------
# -------------------------
# Exctraction S1 2025 :
# ONEDRIVE_URL = "https://adventplus-my.sharepoint.com/:u:/g/personal/igotni_adv-sud_fr/EQ0stgyyDlREjlnsVaUZ01sBQZEoWnscCqhuPEM5iPEmcQ?download=1" #old
ONEDRIVE_URL = "https://adventplus-my.sharepoint.com/:u:/g/personal/igotni_adv-sud_fr/Ef8LL-Y_mNhOlCQlKHlQs1wBXzoorlA-dVNmoZ07zj3oNw?download=1"

# -------------------------

# Extraction TR S1 2025 :

# ONEDRIVE_URL = "https://adventplus-my.sharepoint.com/:u:/g/personal/igotni_adv-sud_fr/EVAEu6MEKhVOqn3UhLlYSyEBNOF9OuzIaUxNd0zjqFLqaw?download=1"
# -------------------------

# Traitement
# -------------------------
def nettoyer_nom(nom):
    """Nettoie les noms de dossiers/fichiers pour compatibilité cross-OS."""
    return re.sub(r'[<>:"/\\|?*]', "_", str(nom).strip()).lower()
from collections import defaultdict
import zipfile, os, re
from io import BytesIO

def normaliser_ref(x) -> str:
    s = str(x).strip()
    try:
        s = s.replace(",", ".")
        f = float(s); i = int(f)
        if f == float(i):
            return str(i)
    except Exception:
        pass
    if "." in s:
        s = s.split(".", 1)[0]
    return s

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
        outer_zip_path = os.path.join(temp_dir, "S1 2025 extract NV.zip")
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
        # Build receipts index directly from the inner ZIP (handles subfolders)
        st.session_state["receipts_index"] = build_receipts_index_from_zipfile(inner_zip_path)
        st.success(f"🔍 Index justificatifs construit pour {len(st.session_state['receipts_index'])} références.")
        st.info(f"📄 Total fichiers justificatifs: {sum(len(v) for v in st.session_state['receipts_index'].values())}")

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
        missions_selected = [nettoyer_nom(m) for m in missions_selected]

        # --- Création des dossiers missions ---
        for mission, group in grouped:
            mission_clean = nettoyer_nom(mission)
            mission_path = os.path.join(temp_dir, mission_clean)
            os.makedirs(mission_path, exist_ok=True)

            # Sauvegarde du rapport Excel
            excel_path = os.path.join(mission_path, f"{mission_clean}.xlsx")
            group.to_excel(excel_path, index=False)

            justificatifs_dir = os.path.join(temp_dir, "justifs")

            for _, row in group.iterrows():
                ref = str(row["Référence"]).strip()
                user_name = nettoyer_nom(row["Utilisateur"])

                # --- Nouveau : créer un dossier par mois ---
                if "Date" in df.columns:
                    try:
                        date_val = pd.to_datetime(row["Date"], errors="coerce")
                        mois_str = date_val.strftime("%B %Y") if pd.notna(date_val) else "inconnu"
                    except Exception:
                        mois_str = "inconnu"
                else:
                    mois_str = "inconnu"

                mois_dir = os.path.join(mission_path, mois_str)
                user_dir = os.path.join(mois_dir, user_name)
                os.makedirs(user_dir, exist_ok=True)

                # Copier les justificatifs correspondants
                # Copier les justificatifs correspondants
                ref = normaliser_ref(row.get("Référence", ""))
                matching_files = st.session_state.get("receipts_index", {}).get(ref, [])

                for (orig_name, data) in matching_files:
                    nom_depense = str(row.get("Nom de la dépense", "inconnu")).strip()
                    categorie = str(row.get("Catégorie", "inconnu")).strip()
                    try:
                        date_val = pd.to_datetime(row.get("Date"), errors="coerce").strftime("%Y-%m-%d")
                    except Exception:
                        date_val = "inconnu"

                    _, ext = os.path.splitext(orig_name)
                    new_name = f"{ref}_{nom_depense}_{categorie}_{date_val}{ext}"
                    new_name = re.sub(r'[<>:\"/\\|?*]', '_', new_name)

                    with open(os.path.join(user_dir, new_name), "wb") as f:
                        f.write(data)






        output = BytesIO()
        added_files = 0

        with zipfile.ZipFile(output, "w") as zipf:
            for mission in missions_selected:
                mission_clean = nettoyer_nom(mission)
                mission_dir = os.path.join(temp_dir, mission_clean)

                if os.path.exists(mission_dir):
                    for root, _, files in os.walk(mission_dir):
                        for file in files:
                            full_path = os.path.join(root, file)
                            rel_path = os.path.relpath(full_path, temp_dir)
                            zipf.write(full_path, rel_path)
                            added_files += 1
                else:
                    st.write(f"🚫 Mission non trouvée : {mission_clean}")

        if added_files > 0:
            output.seek(0)
            st.success("✅ Traitement terminé, toutes vos missions sont prêtes.")
            st.download_button("📥 Télécharger toutes vos missions", output, file_name=f"{user}_missions.zip")
        else:
            st.warning("⚠️ Aucun dossier trouvé pour votre compte.")

            st.write("Missions sélectionnées:", missions_selected)
            st.write("Dossiers générés:", os.listdir(temp_dir))




    except Exception as e:
        
        st.error(f"❌ Erreur : {e}")
# =============================
#     📅 CALENDRIER DES JUSTIFICATIFS
# =============================
st.divider()
st.subheader("📅 Calendrier des justificatifs (par date)")

st.markdown(
    "- Charge le **Rapport Expensya (.xlsx)** (la même feuille 'Rapport' que tu utilises).  \n"
    "- Le graphique affiche chaque dépense à sa date, colorée par **Catégorie**.  \n"
    "- Filtrage automatique sur tes missions sélectionnées."
)

cal_file = st.file_uploader("Importer le fichier Rapport (.xlsx)", type=["xlsx"], key="cal_uploader")

if cal_file is not None:
    try:
        cal_df = pd.read_excel(cal_file, sheet_name="Rapport")

        # Colonnes minimales requises
        required_cols = ["Date", "Nom de la dépense", "Catégorie", "Utilisateur", "Client (Référence)"]
        missing = [c for c in required_cols if c not in cal_df.columns]
        if missing:
            st.error(f"Colonnes manquantes dans le Rapport pour le calendrier : {missing}")
        else:
            # Normalisation
            cal_df["Date"] = pd.to_datetime(cal_df["Date"], errors="coerce")
            cal_df = cal_df.dropna(subset=["Date"])
            cal_df["Client (Référence)"] = cal_df["Client (Référence)"].astype(str)

            # Filtre sur les missions choisies dans l'appli (mêmes valeurs qu’en haut)
            missions_set = set([m.strip().lower() for m in st.session_state.get("missions", [])])
            # Dans ton flux, tu utilises "Mission_Final" plus tard. Ici on se cale sur la colonne d’origine :
            cal_df = cal_df[cal_df["Client (Référence)"].str.lower().isin(missions_set) | (len(missions_set) == 0)]

            if cal_df.empty:
                st.warning("Aucune dépense trouvée pour les missions sélectionnées dans ce fichier.")
            else:
                # Quelques métriques utiles
                col1, col2, col3 = st.columns(3)
                col1.metric("Dépenses", f"{len(cal_df):,}".replace(",", " "))
                col2.metric("Utilisateurs uniques", cal_df["Utilisateur"].nunique())
                col3.metric("Jours distincts", cal_df["Date"].dt.date.nunique())

                # Sélecteurs
                with st.expander("🎛️ Filtres"):
                    users = sorted(cal_df["Utilisateur"].dropna().unique().tolist())
                    cats = sorted(cal_df["Catégorie"].dropna().unique().tolist())
                    sel_users = st.multiselect("Utilisateurs", users, default=users)
                    sel_cats = st.multiselect("Catégories", cats, default=cats)
                    date_min = cal_df["Date"].min().date()
                    date_max = cal_df["Date"].max().date()
                    date_range = st.date_input("Période", (date_min, date_max))

                # Application des filtres
                cal_f = cal_df[
                    cal_df["Utilisateur"].isin(sel_users) &
                    cal_df["Catégorie"].isin(sel_cats) &
                    (cal_df["Date"].dt.date >= pd.to_datetime(date_range[0]).date()) &
                    (cal_df["Date"].dt.date <= pd.to_datetime(date_range[-1]).date())
                ]

                if cal_f.empty:
                    st.info("Aucune dépense ne correspond aux filtres actuels.")
                else:
                    # Prépare une “timeline” à la journée (événements d’une seule journée)
                    # On peut colorer par Catégorie et grouper par Utilisateur
                    events = cal_f.copy()
                    events["Début"] = events["Date"].dt.floor("D")
                    events["Fin"] = events["Date"].dt.floor("D") + pd.Timedelta(days=1)

                    # Titre d’info pour le survol
                    # ---------- (OPTIONNEL) Matrice pour libellé mission ----------
                    mat_file = st.file_uploader(
                        "➕ (optionnel) Matrice Expensya (.xlsx) pour afficher le libellé mission",
                        type=["xlsx"],
                        key="mat_for_cal"
                    )

                    # 1) Charger la matrice si fournie -> construire un mapping client -> label
                    client_to_label = {}
                    if mat_file is not None:
                        try:
                            _mat = pd.read_excel(mat_file, sheet_name="Matrice Expensya")
                            if {"Client (Référence)", "Modification Code Expensya"}.issubset(_mat.columns):
                                client_to_label = dict(
                                    zip(_mat["Client (Référence)"].astype(str),
                                        _mat["Modification Code Expensya"].astype(str))
                                )
                        except Exception:
                            pass  # si la matrice n'est pas valide, on ignore

                    # 2) Créer TOUJOURS la colonne MissionLib (même sans matrice)
                    cal_f["MissionLib"] = (
                        cal_f["Client (Référence)"].astype(str)
                            .map(client_to_label)
                            .fillna(cal_f["Client (Référence)"].astype(str))
                    )

                    # ---------- (OPTIONNEL) ZIP des justificatifs pour aperçu & export ----------
                    zip_justifs = st.file_uploader(
                        "➕ (optionnel) ZIP des justificatifs Expensya (celui contenant les PDFs/IMG)",
                        type=["zip"],
                        key="zip_for_cal"
                    )

                    receipts_index = {}   # dict: ref -> list[(filename: str, bytes: b'...')]
                    if zip_justifs is not None:
                        try:
                            import zipfile, os
                            from io import BytesIO
                            with zipfile.ZipFile(zip_justifs) as z:
                                for name in z.namelist():
                                    if name.endswith("/"):  # ignore folders
                                        continue
                                    base = os.path.basename(name)
                                    if not base: 
                                        continue
                                    ref = base.split("_", 1)[0].strip()  # fichiers "REF_..."
                                    try:
                                        data = z.read(name)
                                    except Exception:
                                        continue
                                    receipts_index.setdefault(ref, []).append((base, data))
                            st.success(f"Justificatifs chargés : {sum(len(v) for v in receipts_index.values())} fichier(s).")
                        except Exception as e:
                            st.warning(f"Impossible de lire le ZIP justificatifs : {e}")


    except Exception as e:
        st.error(f"Erreur lors de la construction du calendrier : {e}")
# ---- Remplace tout "timeline" actuel par ce bloc ----

# 1) Construire une ligne par jour/utilisateur (agrégée)
tmp = cal_f.copy()
tmp["Jour"] = tmp["Date"].dt.floor("D")

def to_item_row(r):
    # Texte de détail par ligne
    montant = ""
    if "TTC (EUR)" in r and pd.notna(r["TTC (EUR)"]):
        montant = f" — {r['TTC (EUR)']:.2f} EUR"
    elif "TTC" in r and pd.notna(r["TTC"]):
        montant = f" — {r['TTC']}"
    return f"• {r['Catégorie']} · {r['Nom de la dépense']}{montant}"

g = (
    tmp
    .sort_values(["Utilisateur","Jour","Date"])
    .groupby(["Utilisateur","Jour"], as_index=False)
    .agg(
        count=("Nom de la dépense", "size"),
        cat_unique=("Catégorie", lambda s: list(pd.unique(s))),
        items=("Nom de la dépense", lambda _: None)  # placeholder pour map
    )
)

# “Mixte” si plusieurs catégories ce jour-là
g["Catégorie"] = g["cat_unique"].apply(lambda cats: cats[0] if len(cats)==1 else "Mixte")

# Fabriquer le hover détaillé (avec la Mission)
idx_cols = ["Utilisateur","Jour"]

# on inclut MissionLib pour l'afficher dans le hover
cols_for_hover = ["Catégorie", "Nom de la dépense", "Date", "MissionLib"]
day_rows = tmp[idx_cols + cols_for_hover].copy()

def to_item_row(r):
    montant = ""
    if "TTC (EUR)" in tmp.columns and pd.notna(r.get("TTC (EUR)")):
        montant = f" — {r['TTC (EUR)']:.2f} EUR"
    elif "TTC" in tmp.columns and pd.notna(r.get("TTC")):
        montant = f" — {r['TTC']}"
    return f"• {r['Catégorie']} · {r['Nom de la dépense']}{montant}"

def build_hover(u, d):
    sub = day_rows[(day_rows["Utilisateur"]==u) & (day_rows["Jour"]==d)]
    # missions du jour (unicité)
    missions = sorted(set(sub["MissionLib"].astype(str)))
    mission_txt = " / ".join(missions) if missions else "—"
    # corps (liste des notes)
    lines = [to_item_row(r) for _, r in sub.iterrows()]
    title = f"{u} — {d.strftime('%Y-%m-%d')}<br><b>Mission :</b> {mission_txt}"
    return title + "<br>" + "<br>".join(lines)

g["hover"] = [build_hover(u, d) for u, d in zip(g["Utilisateur"], g["Jour"])]


# Une barre “d'une journée”
g["Début"] = g["Jour"]
g["Fin"] = g["Jour"] + pd.Timedelta(days=1)

# Pattern (bandes) si plusieurs notes de frais
# (Plotly accepte un array pour pattern_shape par point)
g["pattern"] = g["count"].apply(lambda c: "/" if c > 1 else "")

# 2) Tracer la timeline agrégée, colorée par Catégorie
import plotly.express as px
fig = px.timeline(
    g,
    x_start="Début",
    x_end="Fin",
    y="Utilisateur",
    color="Catégorie",
    hover_data={"hover": True, "Début": False, "Fin": False, "Utilisateur": False, "Catégorie": False},
    custom_data=["hover", "count"],
)
# inverser l'axe Y (meilleure lecture)
fig.update_yaxes(autorange="reversed")

# Appliquer le motif (bandes) seulement quand count>1
fig.update_traces(
    hovertemplate="%{customdata[0]}<extra></extra>",
    marker_line_width=0.5,
    marker_line_color="rgba(0,0,0,0.15)",
    selector=dict(type="bar")  # timeline = bar horizontale
)
# IMPORTANT : pattern par point
for trace in fig.data:
    # Pour chaque trace (catégorie), on récupère les indices du sous-ensemble dans g
    # Plotly ne donne pas l'index facilement, on reconstruit un array de patterns
    # basé sur l’ordre d’apparition des points (même ordre que g filtré par Catégorie)
    cat_name = trace.name
    mask = (g["Catégorie"] == cat_name)
    patterns = g.loc[mask, "pattern"].tolist()
    # Si la catégorie n'est pas "Mixte" et a moins d'éléments, on ajuste
    if len(patterns) != len(trace.x):
        # fallback: ne pas crasher, répéter / tronquer
        from itertools import islice, cycle
        patterns = list(islice(cycle(patterns or [""]), 0, len(trace.x)))
    trace.marker.pattern = dict(shape=patterns, fgopacity=0.35, solidity=0.4)

fig.update_layout(
    height=520,
    margin=dict(l=20, r=20, t=30, b=20),
    legend_title_text="Catégorie (Mixte = plusieurs catégories le même jour)",
    xaxis_title="Date",
    yaxis_title="Utilisateur"
)
st.plotly_chart(fig, use_container_width=True)
# ---------- Aperçu & Export des justificatifs (par jour + utilisateur) ----------
with st.expander("📎 Justificatifs (aperçu + export par jour)"):
    if 'g' in locals() and not g.empty:
        # Sélecteurs jour/utilisateur basés sur les points visibles
        jours_dispos = sorted(g["Jour"].dt.strftime("%Y-%m-%d").unique().tolist())
        users_dispos = sorted(g["Utilisateur"].unique().tolist())
        sel_day = st.selectbox("Jour", jours_dispos) if jours_dispos else None
        sel_user = st.selectbox("Utilisateur", users_dispos) if users_dispos else None

        if sel_day and sel_user:
            day_dt = pd.to_datetime(sel_day).date()
            # Sous-ensemble des lignes (cal_f) de ce jour + utilisateur
            sub = cal_f[
                (cal_f["Utilisateur"] == sel_user) &
                (cal_f["Date"].dt.date == day_dt)
            ].copy().sort_values("Date")

            if sub.empty:
                st.info("Aucune note ce jour pour cet utilisateur.")
            else:
                st.markdown(f"**{sel_user} — {sel_day}**  \n_Missions:_ {', '.join(sorted(set(sub['MissionLib'].astype(str))))}")
                st.divider()

                # Préparer un ZIP global "jour" si des justificatifs existent
                import zipfile
                from io import BytesIO
                zip_buffer = BytesIO()
                zip_count = 0

                for i, r in sub.iterrows():
                    ref = str(r.get("Référence", "")).strip()
                    cat = str(r.get("Catégorie", "")).strip()
                    nom_dep = str(r.get("Nom de la dépense", "")).strip()
                    montant = ""
                    if "TTC (EUR)" in sub.columns and pd.notna(r.get("TTC (EUR)")):
                        montant = f"{r['TTC (EUR)']:.2f} EUR"
                    elif "TTC" in sub.columns and pd.notna(r.get("TTC")):
                        montant = f"{r['TTC']}"

                    st.markdown(f"**{cat}** · _{nom_dep}_  —  {montant or '—'}  \nRéf : `{ref}`")

                    files = receipts_index.get(ref, [])
                    if not files:
                        st.caption("Aucun justificatif trouvé pour cette référence (ZIP non fourni ou fichier absent).")
                    else:
                        # Preview + download par fichier
                        cols = st.columns(min(3, len(files)))
                        for j, (fname, data) in enumerate(files):
                            with cols[j % len(cols)]:
                                ext = os.path.splitext(fname)[1].lower()
                                st.write(fname)
                                if ext in [".png", ".jpg", ".jpeg", ".webp"]:
                                    st.image(data, use_column_width=True)
                                # Bouton de téléchargement individuel
                                st.download_button(
                                    "Télécharger",
                                    data=data,
                                    file_name=fname,
                                    mime="application/octet-stream",
                                    key=f"dl_{ref}_{j}"
                                )
                        # Ajout au ZIP global du jour
                        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zf:
                            for fname, data in files:
                                # Chemin lisible dans le ZIP jour
                                zf.writestr(f"{ref}/{fname}", data)
                                zip_count += 1

                    st.markdown("---")

                # Bouton d’export ZIP du jour
                if zip_count > 0:
                    zip_buffer.seek(0)
                    st.download_button(
                        "📦 Télécharger tous les justificatifs de ce jour (ZIP)",
                        data=zip_buffer.getvalue(),
                        file_name=f"justificatifs_{sel_user}_{sel_day}.zip",
                        mime="application/zip",
                        key=f"zip_day_{sel_user}_{sel_day}"
                    )
                else:
                    st.caption("Aucun justificatif à exporter pour cette sélection.")
    else:
        st.info("Le graphique doit contenir des données pour activer cette section.")

# 3) Détails uniquement pour les jours multi-notes
with st.expander("🔎 Détails des jours avec plusieurs notes de frais"):
    multi = g[g["count"] > 1].sort_values(["Utilisateur","Jour"])
    if multi.empty:
        st.write("Aucun jour ne contient plusieurs notes de frais.")
    else:
        for _, row in multi.iterrows():
            u, d = row["Utilisateur"], row["Jour"]
            sub = tmp[(tmp["Utilisateur"]==u) & (tmp["Date"].dt.floor("D")==d)].copy()
            sub = sub.sort_values("Date")
            st.markdown(f"**{u} — {d.strftime('%Y-%m-%d')}**  \n_Catégories:_ {', '.join(sorted(set(sub['Catégorie'])))}  \n_Nombre de notes:_ {len(sub)}")
            for _, r in sub.iterrows():
                montant = ""
                if "TTC (EUR)" in r and pd.notna(r["TTC (EUR)"]):
                    montant = f" — {r['TTC (EUR)']:.2f} EUR"
                elif "TTC" in r and pd.notna(r["TTC"]):
                    montant = f" — {r['TTC']}"
                st.write(f"- **{r['Catégorie']}** · {r['Nom de la dépense']}{montant}")
# Liste par jour (utile façon “agenda”)
with st.expander("🗒️ Vue 'agenda' (liste par jour)"):
    agenda = (
        cal_f.sort_values("Date")
            .assign(Jour=lambda d: d["Date"].dt.strftime("%Y-%m-%d"))
            .groupby("Jour")
    )
    for day, g in agenda:
        st.markdown(f"**{day}**")
        for _, r in g.iterrows():
            montant = ""
            if "TTC (EUR)" in r and pd.notna(r["TTC (EUR)"]):
                montant = f" — {r['TTC (EUR)']:.2f} EUR"
            st.write(
                f"- {r['Utilisateur']} · _{r['Catégorie']}_ · "
                f"**{r['Nom de la dépense']}**{montant}"
            )
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