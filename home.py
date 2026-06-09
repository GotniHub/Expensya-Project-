import streamlit as st

st.set_page_config(page_title="Accueil - Connexion", layout="wide")
st.logo("LOGO.png", icon_image="Logom.png")

st.write("# Welcome, C'est Advent+ Africa! 👋")
st.image("LOGO.png", width=600)

st.title("🔐 Connexion à votre espace")
st.markdown("<p>Accès réservé aux clients utilisateurs ADVENT+</p>", unsafe_allow_html=True)
# 1. Base utilisateurs (login → mission)
# -------------------------
USERS_S2_2024 = {

    "Valrhona SAS": {
        "password": "tR#8bPz4&YqL1mXs",
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
        "password": "xV4!mQ8^bZ1&nWtX",
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

# ==========================
#  Dictionnaire S1 2025
# ==========================

# 1. Base utilisateurs (login → mission)
# -------------------------
USERS_S1_2025 = {

    "Valrhona SAS": {
        "password": "tR#8bPz4&YqL1mXs",
        "missions": [
            "[131157] catman valrhona", #
            "[106709]-valrhona sas rebond bu+ global", #
            "[106710] valrhona sas rebond fsp", #
            "[24685] - encadrement rcm ae", #
            "[238010] sales academy", #
            "[a113413 ] - rebond bu uk", #
            "[s113402] - accompagnement europe du nord", #
            "a113412 projet dollars" #
        ]
    },

    "ADVENTAE PECO": {
        "password": "Fr@5nPw3!YtK8bXs",
        "missions": [
            "[27233]-adventae peco" #

        ]
        
    },
    "VALRHONA INC": {
        "password": "xV4!mQ8^bZ1&nWtX",
        "missions": [
            "teleprospection cadaff 2025" #
        ]
        
    },  
    "VILLARS MAITRE CHOCOLATIER": {
        "password": "H9!vQe3@cZ6uR%wK",
        "missions": [
            "[38331]-villars-export" #

        ]
        
    },
    "ADVENTAE LATAM": {
        "password": "sM2^Lp7Gz#8aXyVf",
        "missions": [
            "[49130]-adventae latam" #

        ]
        
    },

    "Eclair Vuillemier": {
        "password": "uF8#rK2$yV7pXnQs",
        "missions": [
            "[a114001] eclair vuillemier - france eclair" #
 
        ]
        
    },   

    "PROSPECTION": {
        "password": "uF8#rK2$yV7pXnQs",
        "missions": [
            "[a120101] prospection client" #

        ]
        
    }, 

    "SAVENCIA GOURMET": {
        "password": "cD7%yS2&kQ4!zXnM",
        "missions": [
            "[a112806]", #
            "savencia confluence" #
        
        ]
        
    },

    "Sinodis": {
        "password": "xR4!mQ8^bZ1&nKtP",
        "missions": [
            "[p123001] - sinodis adv+" #
        ]
        
    },   
    "Caraman": {
        "password": "xR4!mQ8^bZ1&nKtP",
        "missions": [
            "[p144101] - recette camaran" #
        ]
        
    },   

    "LES CELLIERS D'ORFEE": {
        "password": "xR4!mQ8^bZ1&nKtP",
        "missions": [
            "les celliers d'orfee" #
        ]
        
    },    
    "vide": {
        "password": "0000",
        "missions": ["vide"]
    },
}
# ==========================
#  Dictionnaire S2 2025
# ==========================

# 1. Base utilisateurs (login → mission)
# -------------------------
USERS_S2_2025 = {

    "Valrhona SAS": {
        "password": "tR#8bPz4&YqL1mXs",
        "missions": [
            "[131157] catman valrhona", #
            "[106709]-valrhona sas rebond bu+ global", #
            "[24685] - encadrement rcm ae", #
            "[238010] sales academy", #
            "[a113409]", #
            "[a123402] coaching valrhona uk", # 
            "[s113402] - accompagnement europe du nord", #
        ]
    },

    "PARIANI": {
        "password": "dL%9aQz2^MpR6tYw",
        "missions": [
            "[a114501] pariani structuration marketing & senso" #
        ]
    },

    "ANTIGON": {
        "password": "xP&4bTs7#YqL1mZn",
        "missions": [
            "[a114801] antigon coaching opérationnel" #
        ]
    },

    "PROSPECTION": {
        "password": "Qa!6rVz8@NtP3mXy",
        "missions": [
            "[a120101] prospection client" #
        ]
    },  

    "NUTRITION ET SANTE": {
        "password": "mY#2kPx9&LrT7aWs",
        "missions": [
            "[a114701] audit organisation & politique commerciale food se" #
        ]
    },
    
    "MASDEU": {
    "password": "Bq5&nCz9!Tt4@hWp",
    "missions": [
        "[a113901] masdeu_structuration politique commerciale export", #
        "[a113902]-test développement commercial itinérant ext" #

        ]
    },

    "VALRHONA INC": {
        "password": "xV4!mQ8^bZ1&nWtX",
        "missions": [
            "teleprospection cadaff 2025" #
        ]
        
    },

    "VILLARS MAITRE CHOCOLATIER": {
        "password": "H9!vQe3@cZ6uR%wK",
        "missions": [
            "[38331]-villars-export" #

        ]
        
    },

    "ADVENTAE LATAM": {
        "password": "sM2^Lp7Gz#8aXyVf",
        "missions": [
            "[49130]-adventae latam" #

        ]
        
    },

    "Eclair Vuillemier": {
        "password": "uF8#rK2$yV7pXnQs",
        "missions": [
            "[a114001] eclair vuillemier - france eclair" #
 
        ]
        
    },   

    "ANDROS": {
        "password": "Lp@3xTs7#WqK9bZn",
        "missions": [
            "[a110702] andros - projet spare" #
 
        ]
        
    },   

    "delice & creation export (dcex)": {
        "password": "nW%6tRx2^ZpL8aQm",
        "missions": [
            "delice & creation export (dcex)" #
 
        ]
        
    },   

    "SOCALAIT": {
        "password": "uF8#rK2$yV7pXnQs",
        "missions": [
            "a133201 socalait 2025" #
 
        ]
        
    },  

    "SAVENCIA GOURMET": {
        "password": "cD7%yS2&kQ4!zXnM",
        "missions": [
            "[a112805] upgrade dg global", #
            "[a112806]", #
            "savencia confluence" #
        
        ]
        
    },

    "Sinodis": {
        "password": "xR4!mQ8^bZ1&nKtP",
        "missions": [
            "[p123001] - sinodis adv+" #
        ]
        
    },   

    "Caraman": {
        "password": "Za#7nPw3&YtL1mQx",
        "missions": [
            "[p144101] - recette camaran" #
        ]
        
    },   

    "LES CELLIERS D'ORFEE": {
        "password": "kR^9bPz2!YqL6mXs",
        "missions": [
            "les celliers d'orfee" #
        ]
        
    },    
    "vide": {
        "password": "0000",
        "missions": ["vide"]
    },
}
# ==========================
#  Dictionnaire Thierry Riva S1 2025
# ==========================

# 1. Base utilisateurs (login → mission)
# -------------------------

USERS_TR_S1_2025 = {

    "Valrhona SAS": {
        "password": "tR#8bPz4&YqL1mXs",
        "missions": [
            "[98966]-advs_rebond"
        ]
    },

    "VILLARS MAITRE CHOCOLATIER": {
        "password": "H9!vQe3@cZ6uR%wK",
        "missions": [
            "[38331]-villars-export"

        ]
        
    },
    "ADVENTAE MENA": {
        "password": "sM2^Lp7Gz#8aXyVf",
        "missions": [
            "[41888] - adventae mena"

        ]
        
    },
    "ADVENTAE PECO": {
        "password": "Fr@5nPw3!YtK8bXs",
        "missions": [
            "[27233]-adventae peco"

        ]
        
    },

    "SAVENCIA GOURMET": {
        "password": "cD7%yS2&kQ4!zXnM",
        "missions": [
            "savencia confluence"
        
        ]
        
    },

    "Sinodis": {
        "password": "xR4!mQ8^bZ1&nKtP",
        "missions": [
            "[24030]-sinodis advs"
        ]
        
    },   
      
    "vide": {
        "password": "0000",
        "missions": ["vide"]
    },
}

# ==========================
#  Dictionnaire Thierry Riva S1 2025
# ==========================

# 1. Base utilisateurs (login → mission)
# -------------------------

USERS_TR_S2_2025 = {

    "Valrhona SAS": {
        "password": "tR#8bPz4&YqL1mXs",
        "missions": [
            "[98966]-cfl_advs_management stratégique" #
        ]
    },


    "ADVENTAE PECO": {
        "password": "Fr@5nPw3!YtK8bXs",
        "missions": [
            "[27233]-ae_peco" #

        ]
        
    },

    "SAVENCIA GOURMET": {
        "password": "cD7%yS2&kQ4!zXnM",
        "missions": [
            "savencia confluence"
        
        ]
        
    },

    "Sinodis": {
        "password": "xR4!mQ8^bZ1&nKtP",
        "missions": [
            "[24030]-sinodis advs" #
        ]
        
    },   

    "ADVENTAE LATAM": {
        "password": "sM2^Lp7Gz#8aXyVf",
        "missions": [
            "[49130]-adventae latam" 

        ]
        
    },      

    "Valrhona SAS": {
        "password": "tR#8bPz4&YqL1mXs",
        "missions": [
            "[106709]-valrhona sas rebond bu+ global", #
            "[238010] sales academy", #
            "[98966]-cfl_advs_management stratégique"
        ]
    },

    "ANDROS": {
        "password": "Lp@3xTs7#WqK9bZn",
        "missions": [
            "[a110702] andros - projet spare" #
 
        ]
        
    },    

    "PARIANI": {
        "password": "dL%9aQz2^MpR6tYw",
        "missions": [
            "[a114501] pariani structuration marketing & senso", #
            "[a114502] pariani - andrea coaching" #
        ]
    }, 

    "PROSPECTION": {
        "password": "uF8#rK2$yV7pXnQs",
        "missions": [
            "[a120101] prospection client" #

        ]
        
    }, 

    "INTERNE": {
        "password": "uFDGhskpkV7pXnFt",
        "missions": [
            "[ac80907] internal teams meeting" #

        ]
        
    }, 

    "AU CHAI VOUS": {
        "password": "uF8#rK2$yV7pXnQs",
        "missions": [
            "acv090001 au chai vous" #

        ]
        
    }, 

    "LES CELLIERS D'ORFEE": {
        "password": "xR4!mQ8^bZ1&nKtP",
        "missions": [
            "les celliers d'orfee" #
        ]
        
    },  

    "VALRHONA INC": {
        "password": "xV4!mQ8^bZ1&nWtX",
        "missions": [
            "teleprospection cadaff 2025"
        ]
        
    },    

    "vide": {
        "password": "0000",
        "missions": ["vide"]
    },
}
USERS_S1_2026 = {

    "MASDEU": {
    "password": "Bq5&nCz9!Tt4@hWp",
    "missions": [
    
        "[a113902]-test développement commercial itinérant ext" #

        ]
    },

    "Advent+": {
        "password": "adv#8bPz4&YqL1mXs",
        "missions": [
            "adv+_formation interne" #
        ]
    },

    "Valrhona SAS": {
        "password": "tR#8bPz4&YqL1mXs",
        "missions": [
            "cfl_bu europ s",
            "cfl_bu france", #
            "cfl_ingeniering",
            "sales academy",
            "trst_cat man mb",
            "wsa_ingé_fsp"
        ]
    },

    "Eclair Vuillemier": {
        "password": "uF8#rK2$yV7pXnQs",
        "missions": [
            "eclair vuilleumier"

        ]
        
    }, 

    "NUTRITION ET SANTE": {
        "password": "mY#2kPx9&LrT7aWs",
        "missions": [
            "nutrition & santé mission transition" #
        ]
    },

    "Partner+": {
        "password": "pt@3xTs7#WqK9bZn",
        "missions": [
            "prj_formation wsa 2.0" #
 
        ]
        
    }, 

    "Sinodis": {
        "password": "xR4!mQ8^bZ1&nKtP",
        "missions": [
            "cfl_bu chine [sinodis]" #
        ]
        
    },   

    "ADVENTAE LATAM": {
        "password": "sM2^Lp7Gz#8aXyVf",
        "missions": [
            "ae_latam" 

        ]
        
    },      

    "ANDROS": {
        "password": "Lp@3xTs7#WqK9bZn",
        "missions": [
            "andros projet spare" #
 
        ]
        
    },    

    "PARIANI": {
        "password": "dL%9aQz2^MpR6tYw",
        "missions": [
            "bud maison [pariani]",
            "pariani – andrea coaching"
            #
        ]
    }, 

    "PROSPECTION": {
        "password": "uF8#rK2$yV7pXnQs",
        "missions": [
            "prospection client" #

        ]
        
    }, 

    "INTERNE": {
        "password": "uFDGhskpkV7pXnFt",
        "missions": [
            "marketing opérationnel adventae" #

        ]
        
    }, 
 
    "vide": {
        "password": "0000",
        "missions": ["vide"]
    },
}
USERSAF_S1_2026 = {


    "Valrhona SAS": {
        "password": "tR#8bPz4&YqL1mXs",
        "missions": [
            "cfl_bu france", #
        ]
    },
 

    "PARIANI": {
        "password": "dL%9aQz2^MpR6tYw",
        "missions": [
            "bud maison [pariani]",
            #
        ]
    }, 

    "SAVENCIA GOURMET": {
        "password": "cD7%yS2&kQ4!zXnM",
        "missions": [
            "[a112806]"
        
        ]
        
    },

    "vide": {
        "password": "0000",
        "missions": ["vide"]
    },
}
USERSLG_S1_2026 = {


    "Eclair Vuillemier": {
        "password": "uF8#rK2$yV7pXnQs",
        "missions": [
            "eclair vuilleumier" #
 
        ]
        
    },

}
USERS_TR_S1_2026 = {

    "SAVENCIA GOURMET": {
        "password": "cD7%yS2&kQ4!zXnM",
        "missions": [
            "[a112805] upgrade dg global", #      
        ]
        
    },    
    "VILLARS MAITRE CHOCOLATIER": {
        "password": "H9!vQe3@cZ6uR%wK",
        "missions": [
            "[38331]-villars-export"

        ]
        
    },
    "Advent+": {
        "password": "adv#8bPz4&YqL1mXs",
        "missions": [
            "adv+_man tr", #
            "adv+_marketing global",
            "mod_senso"
        ]
    },

    "ADVENTAE GLOBAL": {
        "password": "SXFG^Lp7Gz#8a552vs",
        "missions": [
            "ae_groupe" 

        ]
    },     

    "ADVENTAE MENA": {
        "password": "sM2^Lp7Gz#8aXyVf",
        "missions": [
            "ae_mena"

        ]
        
    },

    "ADVENTAE PECO": {
        "password": "Fr@5nPw3!YtK8bXs",
        "missions": [
            "ae_peco" #

        ]
        
    },

    "AU CHAI VOUS": {
        "password": "uF8#rK2$yV7pXnQs",
        "missions": [
            "au chai vous" #

        ]
        
    },  

    "Valrhona SAS": {
        "password": "tR#8bPz4&YqL1mXs",
        "missions": [
            "cfl_advs_management stratégique", #
            "cfl_bu europ s",

        ]
    },
   

    "ADVENTAE LATAM": {
        "password": "sM2^Lp7Gz#8aXyVf",
        "missions": [
            "ae_latam" 

        ]
        
    },        

    "PARIANI": {
        "password": "dL%9aQz2^MpR6tYw",
        "missions": [
            "bud maison [pariani]",
            "pariani – andrea coaching"
            #
        ]
    }, 

    "PROSPECTION": {
        "password": "uF8#rK2$yV7pXnQs",
        "missions": [
            "prospection client" #

        ]
        
    }, 
 
    "vide": {
        "password": "0000",
        "missions": ["vide"]
    },
}
# ==========================
#  Périodes & liens OneDrive
# ==========================
PERIODS = {
    "Consultants internes S2 2024 ": {
        "users": USERS_S2_2024,
        "onedrive_url": "https://adventplus-my.sharepoint.com/:u:/g/personal/igotni_adv-sud_fr/EahoQ8gXXhJLpKJy4FtfyvsBsKc7r60cII0KbVjkorzH6g?download=1"
    },
    "Consultants internes S1 2025": {
        "users": USERS_S1_2025,
        "onedrive_url": "https://adventplus-my.sharepoint.com/:u:/g/personal/igotni_adv-sud_fr/Ef8LL-Y_mNhOlCQlKHlQs1wBXzoorlA-dVNmoZ07zj3oNw?download=1"
    },
    "THIERRY RIVA S1 2025": {
        "users": USERS_TR_S1_2025,
        "onedrive_url": "https://adventplus-my.sharepoint.com/:u:/g/personal/igotni_adv-sud_fr/EVAEu6MEKhVOqn3UhLlYSyEBNOF9OuzIaUxNd0zjqFLqaw?download=1"
    },
    "Consultants internes S2 2025": {
        "users": USERS_S2_2025,
        "onedrive_url": "https://adventplus-my.sharepoint.com/:u:/g/personal/igotni_adv-sud_fr/IQBDtBI36XJ9Q57VYGvpCJYEAXP-k_P5BIT5ICeTQalqUH8?download=1"
    },
    "THIERRY RIVA S2 2025": {
        "users": USERS_TR_S2_2025,
        "onedrive_url": "https://adventplus-my.sharepoint.com/:u:/g/personal/igotni_adv-sud_fr/IQDXgO5qoprSTYwOkua8rfuEAavnZpC9OqNa8ZoaOEt75SE?download=1"
    },
    "Consultants internes S1 trimestre 1 2026": {   
        "users": USERS_S1_2026,  # réutilise les mêmes utilisateurs que S1 2025
        "onedrive_url": "https://adventplus-my.sharepoint.com/:u:/g/personal/igotni_adv-sud_fr/IQCSFmh7-IGSSIWTj1D95esxATcdfVEe6lBd3rSRnKvouWM?download=1"
    },
    "Bureau maroc S1 trimestre 1 2026": {   
        "users": USERSAF_S1_2026,  # réutilise les mêmes utilisateurs que S1 2025
        "onedrive_url": "https://adventplus-my.sharepoint.com/:u:/g/personal/igotni_adv-sud_fr/IQBqhOHg6QzURZ3Ngb2475nLAZSUGYTQvdhg3MjxM3iURaM?download=1"
    },
    "Lionel gerfraud Eclair Vuilleumuier s1 2026": {   
        "users": USERSLG_S1_2026,  # réutilise les mêmes utilisateurs que S1 2025
        "onedrive_url": "https://adventplus-my.sharepoint.com/:u:/g/personal/igotni_adv-sud_fr/IQAD6Fh4bUXCSpjxTTNQuypAAcZmRqPio5VwZGXissfcH3E?download=1"
    },    
    "THIERRY RIVA S1 trimestre 1 2026": {
        "users": USERS_TR_S1_2026,
        "onedrive_url": "https://adventplus-my.sharepoint.com/:u:/g/personal/igotni_adv-sud_fr/IQCkbqMuo_L8T6g67GGlaa3GAdDeq0UR7OXk-cMaZ1wHxOI?download=1"
    },
}

# Initialiser session
if "auth_user" not in st.session_state:
    st.session_state["auth_user"] = None
if "missions" not in st.session_state:
    st.session_state["missions"] = []

if "current_period" not in st.session_state:
    st.session_state["current_period"] = "Consultants internes S1 2025"  # période par défaut
if "onedrive_url" not in st.session_state:
    st.session_state["onedrive_url"] = None

# Choix de la période Expensya (S1 / S2)
period_names = list(PERIODS.keys())
default_idx = period_names.index(st.session_state["current_period"]) \
    if st.session_state["current_period"] in period_names else 0

period_choice = st.selectbox(
    "📁 Période de données Expensya",
    period_names,
    index=default_idx
)

st.session_state["current_period"] = period_choice

# Formulaire login
username = st.text_input("Identifiant")
password = st.text_input("Mot de passe", type="password")

if st.button("🔑 Se connecter"):
    # Choisir le dictionnaire USERS correspondant à la période sélectionnée
    period_cfg = PERIODS[st.session_state["current_period"]]
    USERS = period_cfg["users"]

    if username in USERS and USERS[username]["password"] == password:
        
        st.session_state["auth_user"] = username
        st.session_state["missions"] = USERS[username]["missions"]
        st.session_state["onedrive_url"] = period_cfg["onedrive_url"]

        st.success(f"Bienvenue {username} 👋 (période : {st.session_state['current_period']})")
        st.switch_page("pages/Justificatifs local.py")
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
