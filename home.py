import streamlit as st

st.set_page_config(page_title="Accueil - Connexion", layout="wide")
st.logo("LOGO.png", icon_image="Logom.png")

st.write("# Welcome, C'est Advent+ Africa! 👋")
st.image("LOGO.png", width=600)

st.title("🔐 Connexion à votre espace")
st.markdown("<p>Accès réservé aux clients utilisateurs ADVENT+</p>", unsafe_allow_html=True)
# 1. Base utilisateurs (login → mission)
# -------------------------
USERS = {

    "Valrhona SAS": {
        "password": "1234",
        "missions": [
            "[131157] catman valrhona",
            "[106709]-valrhona sas rebond bu+ global",
            "[106710] valrhona sas rebond fsp",
            "[24685] - encadrement rcm ae",
            "[238010] sales academy",
            "top order [131156]"
        ]
    },
    "VALRHONA INC": {
        "password": "1234",
        "missions": [
            "teleprospection cadaff 2025"
        ]
        
    },  
    "VILLARS MAITRE CHOCOLATIER": {
        "password": "H9!vQe3@cZ6uR%wK",
        "missions": [
            "[38331]-villars-export"

        ]
        
    },
    "ADVENTAE LATAM": {
        "password": "sM2^Lp7Gz#8aXyVf",
        "missions": [
            "[49130]-adventae latam"

        ]
        
    },
    "MASDEU": {
        "password": "Bq5&nCz9!Tt4@hWp",
        "missions": [
            "[a113901] masdeu_structuration politique commerciale export"

        ]
        
    },
    "Eclair Vuillemier": {
        "password": "uF8#rK2$yV7pXnQs",
        "missions": [
            "[a114001] eclair vuillemier - france eclair"

        ]
        
    },   
    "SAVENCIA FROMAGE & DAIRY US": {
        "password": "Zp3!wH6@bR9^mLvT",
        "missions": [
            "[a141401] qualif savencia us 2024",
        ]
        
    },
    "SAVENCIA GOURMET": {
        "password": "cD7%yS2&kQ4!zXnM",
        "missions": [
            "[131155] portage fdv  apero food  service",
            "savencia confluence"
        
        ]
        
    },
    "LESAFFRE": {
        "password": "Nw6#tV9$gR3@pHyL",
        "missions": [
            "[235030] - lesaffre audit orga com canal indirect"
        
        ]
        
    },
    "Sinodis": {
        "password": "xR4!mQ8^bZ1&nKtP",
        "missions": [
            "[p123001] - sinodis adv+"
        ]
        
    },   
      
    "vide": {
        "password": "0000",
        "missions": ["vide"]
    },
}

# Initialiser session
if "auth_user" not in st.session_state:
    st.session_state["auth_user"] = None
if "missions" not in st.session_state:
    st.session_state["missions"] = []
# Formulaire login
username = st.text_input("Identifiant")
password = st.text_input("Mot de passe", type="password")

if st.button("Se connecter"):
    if username in USERS and USERS[username]["password"] == password:
        st.session_state["auth_user"] = username
        st.session_state["missions"] = USERS[username]["missions"]
        st.success(f"Bienvenue {username} 👋")
        st.switch_page("pages/Justificatifs.py")  # redirection vers App
    else:
        st.error("❌ Identifiants incorrects")
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
