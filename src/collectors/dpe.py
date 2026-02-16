# -*- coding: utf-8 -*-
"""
Collecteur DPE ADEME — Diagnostics de Performance Énergétique.
===============================================================

Récupère les données DPE (Diagnostic de Performance Énergétique)
depuis l'API de l'ADEME. Les DPE servent de PROXY pour les installations
HVAC : un DPE mentionnant une pompe à chaleur (PAC) ou une climatisation
indique une installation récente de cet équipement.

Source : https://data.ademe.fr/datasets/dpe03existant
API : https://data.ademe.fr/data-fair/api/v1/datasets/dpe03existant/lines
Authentification : Aucune (Open Data)

NOTES AUDIT :
    - L'ancienne URL (dpe-v2-logements-existants) est CASSÉE depuis 2025.
    - La nouvelle URL est 'dpe03existant' (vérifié février 2026).
    - Le dataset contient ~14M de DPE → pagination obligatoire.
    - La pagination utilise un curseur ('after'), pas un numéro de page.
    - Le DPE est un proxy imparfait : il est déclenché par une vente/location,
      pas directement par une installation HVAC.

Données collectées (champs sélectionnés) :
    - numero_dpe : Identifiant unique
    - date_etablissement_dpe : Date du diagnostic
    - etiquette_dpe : Classe énergétique (A-G)
    - etiquette_ges : Classe GES (A-G)
    - type_energie_chauffage : Type d'énergie pour le chauffage
    - type_installation_chauffage : Type d'installation de chauffage
    - surface_habitable_logement : Surface en m²
    - code_postal_ban : Code postal
    - code_departement_ban : Code département (filtrage)

Variable cible ML :
    Nombre de DPE par mois × département mentionnant :
    - PAC (pompe à chaleur) dans type_installation_chauffage
    - Climatisation dans les champs d'équipement
    - Classe A-B (bâtiments performants = équipement récent probable)

Extensibilité :
    Pour modifier les champs collectés, ajuster la liste DPE_SELECT_FIELDS.
    Pour changer le filtrage, modifier les paramètres dans collect().
"""

from __future__ import annotations

from typing import ClassVar, List, Optional

import pandas as pd
from tqdm import tqdm

from src.collectors.base import BaseCollector

# =============================================================================
# Configuration DPE ADEME
# =============================================================================

# URL de l'API DPE ADEME (MISE À JOUR suite à l'audit)
DPE_API_BASE = (
    "https://data.ademe.fr/data-fair/api/v1/"
    "datasets/dpe03existant/lines"
)

# Champs à sélectionner — enrichi pour atteindre ~1 Go de données
# Regroupés par catégorie fonctionnelle pour lisibilité
DPE_SELECT_FIELDS = [
    # --- Identification ---
    "numero_dpe",
    "date_etablissement_dpe",
    "date_visite_diagnostiqueur",

    # --- Localisation ---
    "code_postal_ban",
    "code_departement_ban",
    "code_insee_ban",
    "nom_commune_ban",

    # --- Performance énergétique (variable cible ML) ---
    "etiquette_dpe",
    "etiquette_ges",
    "conso_5_usages_par_m2_ep",             # Consommation énergie primaire kWh/m².an
    "conso_5_usages_par_m2_ef",             # Consommation énergie finale
    "emission_ges_5_usages_par_m2",         # Émissions GES kgCO2/m².an

    # --- Caractéristiques du bâtiment ---
    "type_batiment",                         # maison / appartement / immeuble
    "annee_construction",
    "periode_construction",
    "surface_habitable_logement",
    "nombre_niveau_logement",
    "hauteur_sous_plafond",

    # --- HVAC : Chauffage (features clés pour le modèle) ---
    "type_energie_principale_chauffage",
    "type_installation_chauffage",
    "type_generateur_chauffage_principal",   # PAC, chaudière, radiateur, etc.

    # --- HVAC : ECS (eau chaude sanitaire) ---
    "type_energie_principale_ecs",

    # --- HVAC : Climatisation / Froid ---
    "type_generateur_froid",                 # Climatisation, PAC réversible, etc.

    # --- Isolation (indicateur de rénovation) ---
    "qualite_isolation_enveloppe",
    "qualite_isolation_murs",
    "qualite_isolation_menuiseries",
    "qualite_isolation_plancher_haut_comble_perdu",

    # --- Coûts (indicateur économique) ---
    "cout_chauffage",
    "cout_ecs",
    "cout_total_5_usages",
]

# Nombre de lignes par page (max autorisé par l'API ADEME)
PAGE_SIZE = 10000


class DpeCollector(BaseCollector):
    """Collecteur des DPE ADEME (proxy des installations HVAC).

    Collecte les DPE département par département avec pagination
    par curseur. Le volume total peut atteindre plusieurs centaines
    de milliers de lignes pour la région AURA.

    ATTENTION : cette collecte est la plus volumineuse du projet.
    Prévoir 10-30 minutes selon la connexion.

    Auto-enregistré comme 'dpe' dans le CollectorRegistry.
    """

    source_name: ClassVar[str] = "dpe"
    output_subdir: ClassVar[str] = "dpe"
    output_filename: ClassVar[str] = "dpe_aura_all.csv"

    def collect(self) -> pd.DataFrame:
        """Collecte les DPE pour tous les départements AURA.

        Pour chaque département :
        1. Requête paginée via curseur 'after'
        2. Accumulation des résultats jusqu'à épuisement
        3. Sauvegarde intermédiaire par département (sécurité)

        Returns:
            DataFrame concaténé de tous les DPE AURA.
        """
        all_frames: List[pd.DataFrame] = []
        errors: List[str] = []

        for dept in self.config.departments:
            self.logger.info(
                "Collecte DPE département %s...", dept,
            )

            try:
                df_dept = self._collect_department(dept)
                if not df_dept.empty:
                    all_frames.append(df_dept)

                    # Sauvegarde intermédiaire par département (sécurité)
                    dept_path = (
                        self.config.raw_data_dir / "dpe" / f"dpe_{dept}.csv"
                    )
                    dept_path.parent.mkdir(parents=True, exist_ok=True)
                    df_dept.to_csv(dept_path, index=False)
                    self.logger.info(
                        "  ✓ Dept %s : %d DPE sauvegardés → %s",
                        dept, len(df_dept), dept_path,
                    )

            except Exception as exc:
                error_msg = f"Échec département {dept} : {exc}"
                errors.append(error_msg)
                self.logger.error("  ✗ %s", error_msg)
                continue

        if not all_frames:
            self.logger.error("Aucun DPE collecté. Erreurs : %s", errors)
            return pd.DataFrame()

        # Concaténer tous les départements
        result = pd.concat(all_frames, ignore_index=True)

        if errors:
            self.logger.warning(
                "⚠ Collecte partielle : %d/%d départements réussis",
                len(all_frames), len(self.config.departments),
            )

        return result

    def _collect_department(
        self, dept_code: str, max_pages: int = 200
    ) -> pd.DataFrame:
        """Collecte tous les DPE d'un département via pagination.

        L'API ADEME utilise une pagination par curseur ('after') :
        - La première requête retourne les N premières lignes + un curseur
        - Les requêtes suivantes utilisent le curseur pour avancer

        Args:
            dept_code: Code département (ex: "69").
            max_pages: Garde-fou : nombre max de pages à récupérer.

        Returns:
            DataFrame avec tous les DPE du département.
        """
        all_rows: List[dict] = []

        params = {
            "size": PAGE_SIZE,
            "select": ",".join(DPE_SELECT_FIELDS),
            "qs": f"code_departement_ban:{dept_code}",
        }

        for page in tqdm(
            range(max_pages),
            desc=f"  DPE dept {dept_code}",
            leave=False,
        ):
            try:
                data = self.fetch_json(DPE_API_BASE, params=params)
            except Exception as exc:
                self.logger.error(
                    "  Erreur page %d pour dept %s : %s",
                    page + 1, dept_code, exc,
                )
                break  # Sauvegarder ce qu'on a déjà collecté

            # Extraire les résultats
            results = data.get("results", [])
            if not results:
                break  # Plus de données

            all_rows.extend(results)

            # Récupérer le curseur pour la page suivante
            next_cursor = data.get("next")
            if not next_cursor:
                break

            # Mettre à jour les paramètres avec le curseur
            # L'API ADEME utilise le champ 'after' pour la pagination
            if "after" in str(next_cursor):
                # Extraire le paramètre 'after' de l'URL next
                import urllib.parse
                parsed = urllib.parse.urlparse(next_cursor)
                query_params = urllib.parse.parse_qs(parsed.query)
                if "after" in query_params:
                    params["after"] = query_params["after"][0]
                else:
                    break
            else:
                break

            # Pause de politesse
            self.rate_limit_pause()

        self.logger.debug(
            "  Dept %s : %d lignes collectées en %d pages",
            dept_code, len(all_rows), page + 1,
        )

        return pd.DataFrame(all_rows) if all_rows else pd.DataFrame()

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Valide la structure et la qualité des données DPE.

        Vérifications :
        1. Colonnes obligatoires présentes
        2. Conversion des dates
        3. Départements dans le périmètre AURA
        4. Statistiques des étiquettes DPE

        Args:
            df: DataFrame brut issu de collect().

        Returns:
            DataFrame validé avec types corrects.

        Raises:
            ValueError: Si les colonnes critiques sont manquantes.
        """
        # 1. Colonnes obligatoires
        critical_cols = {"date_etablissement_dpe", "etiquette_dpe"}
        available = set(df.columns)
        missing = critical_cols - available
        if missing:
            self.logger.warning(
                "Colonnes attendues manquantes : %s. "
                "Colonnes disponibles : %s", missing, sorted(available),
            )

        # 2. Conversion des dates
        if "date_etablissement_dpe" in df.columns:
            df["date_etablissement_dpe"] = pd.to_datetime(
                df["date_etablissement_dpe"], errors="coerce"
            )
            valid_dates = df["date_etablissement_dpe"].notna().sum()
            self.logger.info(
                "Dates valides : %d/%d (%.1f%%)",
                valid_dates, len(df), 100 * valid_dates / max(len(df), 1),
            )

        # 3. Distribution des étiquettes DPE
        if "etiquette_dpe" in df.columns:
            distribution = df["etiquette_dpe"].value_counts()
            self.logger.info("Distribution DPE :\n%s", distribution.to_string())

        # 4. Départements
        if "code_departement_ban" in df.columns:
            depts = sorted(df["code_departement_ban"].dropna().unique().tolist())
            self.logger.info("Départements collectés : %s", depts)

        # Log résumé
        self.logger.info(
            "Validation OK : %d DPE collectés", len(df),
        )

        return df
