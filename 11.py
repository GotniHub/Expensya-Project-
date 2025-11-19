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
    <p style="margin: 0; font-size: 0.9rem; opacity: 0.95;">✔️  Connecté avec succès</p>
</div>
"""
st.sidebar.markdown(card_html, unsafe_allow_html=True)

if st.sidebar.button("🚪 Déconnexion"):
    st.session_state["auth_user"] = None
    st.switch_page("home.py")

# =============================
#     LIEN ONEDRIVE (EXEMPLE)
# =============================
# Exctraction S1 2025 (actuel) :
ONEDRIVE_URL = "https://adventplus-my.sharepoint.com/:u:/g/personal/igotni_adv-sud_fr/Ef8LL-Y_mNhOlCQlKHlQs1wBXzoorlA-dVNmoZ07zj3oNw?download=1"

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

cal_file = st.file_uploader("Importer le fichier Rapport (.xlsx)", type=["xlsx"], key="cal_uploader")

if cal_file is not None:
    try:
        cal_df = pd.read_excel(cal_file, sheet_name="Rapport")

        required_cols = ["Date", "Nom de la dépense", "Catégorie", "Utilisateur", "Client (Référence)", "Référence"]
        missing = [c for c in required_cols if c not in cal_df.columns]
        if missing:
            st.error(f"Colonnes manquantes dans le Rapport pour le calendrier : {missing}")
            st.stop()

        # Normalisation
        cal_df["Date"] = pd.to_datetime(cal_df["Date"], errors="coerce")
        cal_df = cal_df.dropna(subset=["Date"])
        cal_df["Client (Référence)"] = cal_df["Client (Référence)"].astype(str)
        cal_df["Référence"] = cal_df["Référence"].astype(str)

        # Filtre missions (celles choisies en haut de page)
        missions_set = {m.strip().lower() for m in missions_selected}
        if missions_set:
            cal_df = cal_df[cal_df["Client (Référence)"].str.lower().isin(missions_set)]

        if cal_df.empty:
            st.warning("Aucune dépense trouvée pour les missions sélectionnées dans ce fichier.")
            st.stop()

        # Matrice optionnelle (pour afficher un libellé mission lisible)
        mat_file = st.file_uploader(
            "➕ (optionnel) Matrice Expensya (.xlsx) pour afficher le libellé mission",
            type=["xlsx"],
            key="mat_for_cal"
        )
        client_to_label = {}
        if mat_file is not None:
            try:
                _mat = pd.read_excel(mat_file)
                if {"Client (Référence)", "Modification Code Expensya"}.issubset(_mat.columns):
                    client_to_label = dict(
                        zip(_mat["Client (Référence)"].astype(str),
                            _mat["Modification Code Expensya"].astype(str))
                    )
            except Exception:
                pass

        cal_df["MissionLib"] = (
            cal_df["Client (Référence)"].map(client_to_label).fillna(cal_df["Client (Référence)"])
        )

        # ZIP justificatifs optionnel (pour preview / download)
        zip_for_calendar = st.file_uploader(
            "➕ (optionnel) ZIP des justificatifs (export Expensya) — pour prévisualiser/télécharger",
            type=["zip"],
            key="zip_for_calendar"
        )

        # Métriques
        col1, col2, col3 = st.columns(3)
        col1.metric("Dépenses", f"{len(cal_df):,}".replace(",", " "))
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

        # ---------- Vue scatter cliquable (un point = une dépense) ----------
        cal_f["Date_str"] = cal_f["Date"].dt.strftime("%Y-%m-%d")
        cal_f["size"] = 12  # taille fixe des points

        import plotly.express as px
        fig = px.scatter(
            cal_f,
            x="Date", y="Utilisateur",
            color="Catégorie",
            size="size",
            hover_name="Nom de la dépense",
            hover_data={
                "Date": True,
                "Utilisateur": True,
                "Catégorie": True,
                "MissionLib": True,
                "Référence": True,
                "size": False
            },
            custom_data=["Utilisateur", "Date_str", "Référence"],
            color_discrete_sequence=px.colors.qualitative.Set2  # palette sûre (pas monochrome)
        )
        # --- Activer le clic et stabiliser l'état visuel ---
        fig.update_layout(
            height=520,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis_title="Date",
            yaxis_title="Utilisateur",
            legend_title_text="Catégorie",
            clickmode="event+select",   # capter les clics
            uirevision="cal_v1"         # garder l'état visuel entre reruns
        )
        fig.update_traces(mode="markers")  # on clique bien des marqueurs
        # --- Affichage du graphique + capture du clic (ou fallback) ---
        selected = []
        if PLOTLY_EVENTS_AVAILABLE:
            selected = plotly_events(
                fig,
                click_event=True,
                select_event=False,
                hover_event=False,
                override_height=520,
                override_width="100%",
                key="cal_ev"
            )
        else:
            st.plotly_chart(fig, use_container_width=True)
            st.info("Le module 'streamlit-plotly-events' n'est pas disponible ici. Utilise la sélection manuelle ci-dessous.")
            
        # --- État persistant de la sélection ---
        if "cal_selected" not in st.session_state:
            st.session_state["cal_selected"] = None

        # MAJ depuis un clic
        if selected:
            point = selected[-1]
            cd = point.get("customdata") or []
            if len(cd) >= 2:
                st.session_state["cal_selected"] = {"user": cd[0], "day": cd[1]}

        # Fallback manuel si pas de clic
        sel = st.session_state["cal_selected"]
        if not sel:
            st.markdown("#### 🔎 Sélection manuelle (fallback)")
            colu, cold = st.columns([1,1])

            users_opts = sorted(cal_f["Utilisateur"].dropna().unique().tolist())
            user_pick = colu.selectbox("Utilisateur", ["—"] + users_opts, index=0)

            if user_pick != "—":
                day_opts = sorted(
                    cal_f.loc[cal_f["Utilisateur"] == user_pick, "Date"].dt.strftime("%Y-%m-%d").unique().tolist()
                )
                day_pick = cold.selectbox("Jour", ["—"] + day_opts, index=0)
                if day_pick != "—":
                    st.session_state["cal_selected"] = {"user": user_pick, "day": day_pick}

        # bouton reset
        col_reset, _ = st.columns([1,6])
        if col_reset.button("🧹 Effacer la sélection"):
            st.session_state["cal_selected"] = None

        # Récupérer la sélection finale
        sel = st.session_state["cal_selected"]

        # --- État de sélection persistant ---
        if "cal_selected" not in st.session_state:
            st.session_state["cal_selected"] = None  # {"user": ..., "day": ...}

        # (optionnel) bouton pour vider la sélection
        col_reset, _ = st.columns([1, 6])
        if col_reset.button("🔄 Effacer la sélection"):
            st.session_state["cal_selected"] = None

        # Affichage & capture du clic
        # --- Affichage & capture du clic ---
        selected = []
        try:
            from streamlit_plotly_events import plotly_events
            selected = plotly_events(
                fig,
                click_event=True,
                select_event=False,      # clic simple
                hover_event=False,
                override_height=520,
                override_width="100%",
                key="cal_ev"             # clé STABLE
            )
        except Exception:
            st.plotly_chart(fig, use_container_width=True)
            selected = []
            st.caption("DEBUG: plotly_events indisponible")  # (optionnel)

        # --- Mettre à jour la sélection si on a cliqué un point ---
        if selected:
            point = selected[-1]  # le plus récent
            cd = point.get("customdata") or []
            if len(cd) >= 2:
                st.session_state["cal_selected"] = {"user": cd[0], "day": cd[1]}

        sel = st.session_state.get("cal_selected")


        # Index justificatifs si ZIP fourni
        # --------- Accès aux justificatifs ---------
        ref_files = {}
        zf = None

        # 1) Cas A : l'utilisateur a uploadé un ZIP -> priorité
        if zip_for_calendar is not None:
            try:
                zf = zipfile.ZipFile(zip_for_calendar)
                # (re)construit l'index à partir de l'upload
                ref_files = {}
                for name in zf.namelist():
                    base = os.path.basename(name)
                    if not base:
                        continue
                    # même normalisation que plus haut
                    m = re.match(r"^(\d+)[\s_\-\.].*", base) or re.search(r"(\d{3,})", base)
                    if m:
                        key = re.sub(r"\D","", m.group(1)).lstrip("0") or "0"
                        ref_files.setdefault(key, []).append(name)
            except Exception as e:
                st.warning(f"Impossible de lire le ZIP justificatifs uploadé : {e}")
                zf = None

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

        # --- Afficher les détails UNIQUEMENT s'il y a une sélection ---
        if sel:
            sel_user = sel["user"]
            sel_day  = sel["day"]  # "YYYY-MM-DD"

            st.subheader(f"🗂️ Détails pour {sel_user} — {sel_day}")

            clicked_df = cal_f[
                (cal_f["Utilisateur"] == sel_user) &
                (cal_f["Date"].dt.strftime('%Y-%m-%d') == sel_day)
            ].sort_values("Date")

            if clicked_df.empty:
                st.info("Aucune note de frais pour ce point.")
            else:
                import base64, os
                import streamlit.components.v1 as components

                for _, r in clicked_df.iterrows():
                    ref = str(r.get("Référence", "")).strip()
                    mission_txt = r.get("MissionLib", r.get("Client (Référence)", ""))
                    montant = ""
                    if "TTC (EUR)" in r and pd.notna(r["TTC (EUR)"]):
                        montant = f" — {r['TTC (EUR)']:.2f} EUR"

                    with st.expander(
                        f"📄 {r['Catégorie']} · {r['Nom de la dépense']}{montant} · _Mission : {mission_txt}_"
                    ):
                        st.write(f"**Référence** : `{ref}`")
                        st.write(f"**Heure** : {r['Date'].strftime('%H:%M')}")

                        if zf is None:
                            st.caption("🛈 Fournis le ZIP des justificatifs ci-dessus pour prévisualiser et télécharger les fichiers.")
                        else:
                            # 🔧 Normalisation de la référence (pour retrouver les bons fichiers)
                            raw_ref = str(r.get("Référence", "")).strip()
                            norm_ref = re.sub(r"\D", "", raw_ref).lstrip("0") or "0"
                            files = ref_files.get(norm_ref, [])

                            if not files:
                                st.warning("Aucune pièce jointe trouvée pour cette référence.")
                            else:
                                for j, path in enumerate(files):
                                    data = zf.read(path)
                                    name = os.path.basename(path)
                                    ext = name.lower().split(".")[-1]

                                    if ext in ("jpg", "jpeg", "png"):
                                        st.image(data, caption=name, use_container_width=True)
                                        st.download_button("Télécharger l'image", data, file_name=name, key=f"dl_img_{ref}_{j}")
                                    elif ext == "pdf":
                                        b64 = base64.b64encode(data).decode("utf-8")
                                        components.html(
                                            f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="600px" style="border:0;"></iframe>',
                                            height=620, scrolling=True
                                        )
                                        st.download_button("Télécharger le PDF", data, file_name=name, key=f"dl_pdf_{ref}_{j}")
                                    else:
                                        st.download_button(f"Télécharger ({ext})", data, file_name=name, key=f"dl_file_{ref}_{j}")

        else:
            st.info("Clique sur un point pour afficher ses justificatifs (ou utilise les filtres).")


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

    except Exception as e:
        st.error(f"Erreur lors de la construction du calendrier : {e}")


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
