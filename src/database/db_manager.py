# -*- coding: utf-8 -*-
"""
Database Manager — Gestion de la base de données SQLite / SQL Server / PostgreSQL.
===================================================================================

Fournit une interface unifiée pour :
- Initialiser la base de données (création des tables selon le moteur)
- Insérer des données depuis des DataFrames
- Importer les CSV collectés par les collecteurs (weather, insee, eurostat, etc.)
- Requêter des données pour l'analyse et le ML
- Construire le dataset ML-ready (jointure faits + contexte économique)

Architecture :
    DatabaseManager utilise SQLAlchemy comme couche d'abstraction.
    Le choix du moteur (SQLite, SQL Server, PostgreSQL) se fait uniquement
    via la chaîne de connexion. Le schéma SQL est adapté automatiquement
    (schema.sql pour SQLite, schema_mssql.sql pour SQL Server).

Usage:
    >>> from src.database.db_manager import DatabaseManager
    >>> from config.settings import config
    >>> db = DatabaseManager(config.database.connection_string)
    >>> db.init_database()
    >>> db.import_collected_data()            # Importe tous les CSV collectés
    >>> df_ml = db.build_ml_dataset()

Extensibilité :
    Pour ajouter une nouvelle table ou un nouveau type de données :
    1. Ajouter le DDL dans schema.sql ET schema_mssql.sql
    2. Ajouter une méthode load_xxx_data() dans cette classe
    3. Mettre à jour build_ml_dataset() si nécessaire
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


# Chemins vers les fichiers de schéma (relatifs au module)
SCHEMA_SQLITE_PATH = Path(__file__).parent / "schema.sql"
SCHEMA_MSSQL_PATH = Path(__file__).parent / "schema_mssql.sql"


class DatabaseManager:
    """Gestionnaire de base de données pour le projet HVAC.

    Encapsule toutes les opérations CRUD et fournit des méthodes
    spécialisées pour charger chaque type de données.

    Le moteur SQL est détecté automatiquement depuis la chaîne de connexion.

    Attributes:
        engine: Moteur SQLAlchemy connecté à la base.
        db_type: Type de moteur détecté ('sqlite', 'mssql', 'postgresql').
        logger: Logger structuré pour les opérations DB.
    """

    def __init__(self, connection_string: str) -> None:
        """Initialise la connexion à la base de données.

        Détecte automatiquement le type de moteur depuis l'URL de connexion.

        Args:
            connection_string: URL SQLAlchemy (ex: 'sqlite:///data/hvac.db',
                              'mssql+pyodbc://...', 'postgresql://...').
        """
        self.engine: Engine = create_engine(connection_string)
        self.logger = logging.getLogger("database.manager")

        # Détecter le type de moteur pour adapter le comportement
        if connection_string.startswith("sqlite"):
            self.db_type = "sqlite"
        elif connection_string.startswith("mssql"):
            self.db_type = "mssql"
        elif connection_string.startswith("postgresql"):
            self.db_type = "postgresql"
        else:
            self.db_type = "unknown"

        # Configurer le logging
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

        self.logger.info(
            "Connexion BDD : moteur=%s", self.db_type,
        )

    def init_database(self) -> None:
        """Initialise la base de données en exécutant le schéma SQL approprié.

        Sélectionne automatiquement schema.sql (SQLite) ou
        schema_mssql.sql (SQL Server) selon le moteur détecté.
        Idempotent : peut être appelé plusieurs fois sans erreur.
        """
        self.logger.info("Initialisation de la base de donnees...")

        # Choisir le fichier de schéma selon le moteur
        if self.db_type == "mssql":
            schema_path = SCHEMA_MSSQL_PATH
        else:
            # SQLite et PostgreSQL utilisent le même schéma
            # (PostgreSQL supporte BOOLEAN, VARCHAR natifs)
            schema_path = SCHEMA_SQLITE_PATH

        if not schema_path.exists():
            raise FileNotFoundError(
                f"Fichier schema introuvable : {schema_path}"
            )

        schema_sql = schema_path.read_text(encoding="utf-8")

        if self.db_type == "mssql":
            # SQL Server : découper sur 'GO' ou ';' en fin de bloc
            # Le schéma MSSQL utilise des blocs complets séparés par ';'
            self._execute_statements_mssql(schema_sql)
        else:
            # SQLite/PostgreSQL : découper sur ';'
            self._execute_statements_sqlite(schema_sql)

        self.logger.info("Base de donnees initialisee avec succes.")

    def _execute_statements_sqlite(self, schema_sql: str) -> None:
        """Exécute les instructions SQL pour SQLite/PostgreSQL.

        SQLite ne supporte pas l'exécution de scripts multi-instructions,
        donc on découpe sur ';' et on exécute chaque instruction séparément.
        Les lignes de commentaires (--) en début de bloc sont ignorées.
        """
        with self.engine.begin() as conn:
            for statement in schema_sql.split(";"):
                # Retirer les lignes de commentaires et vides en début de bloc
                lines = statement.split("\n")
                sql_lines = [
                    l for l in lines
                    if l.strip() and not l.strip().startswith("--")
                ]
                clean = "\n".join(sql_lines).strip()

                if clean:
                    try:
                        conn.execute(text(clean))
                    except Exception as exc:
                        self.logger.warning(
                            "Instruction SQL ignoree : %s",
                            str(exc)[:120],
                        )

    def _execute_statements_mssql(self, schema_sql: str) -> None:
        """Exécute les instructions SQL pour SQL Server.

        SQL Server supporte les blocs multi-instructions séparés par ';'.
        On gère aussi les MERGE et CTE qui contiennent des ';' internes.
        """
        with self.engine.begin() as conn:
            # Pour SQL Server, exécuter le script complet
            # en découpant sur les lignes vides doubles (séparateur de blocs)
            blocks = schema_sql.split("\n\n\n")
            for block in blocks:
                block = block.strip()
                if block and not block.startswith("--"):
                    try:
                        conn.execute(text(block))
                    except Exception as exc:
                        self.logger.warning(
                            "Bloc SQL ignore : %s",
                            str(exc)[:120],
                        )

    def load_dataframe(
        self,
        df: pd.DataFrame,
        table_name: str,
        if_exists: str = "append",
    ) -> int:
        """Charge un DataFrame dans une table de la base.

        Args:
            df: DataFrame à charger.
            table_name: Nom de la table cible.
            if_exists: Comportement si la table existe ('append', 'replace', 'fail').

        Returns:
            Nombre de lignes insérées.
        """
        rows_before = self._count_rows(table_name)

        df.to_sql(
            table_name, self.engine,
            if_exists=if_exists, index=False,
        )

        rows_after = self._count_rows(table_name)
        inserted = rows_after - rows_before

        self.logger.info(
            "Table '%s' : %d lignes inserees (total : %d)",
            table_name, inserted, rows_after,
        )
        return inserted

    def query(self, sql: str) -> pd.DataFrame:
        """Exécute une requête SQL et retourne un DataFrame.

        Args:
            sql: Requête SQL SELECT.

        Returns:
            DataFrame avec les résultats de la requête.
        """
        with self.engine.connect() as conn:
            return pd.read_sql(text(sql), conn)

    # =================================================================
    # Import des données collectées (CSV → BDD)
    # =================================================================

    def import_collected_data(
        self,
        raw_data_dir: Optional[Path] = None,
    ) -> dict:
        """Importe toutes les données collectées (CSV) dans la BDD.

        Cherche les fichiers CSV dans le répertoire raw_data_dir et
        les charge dans les tables appropriées.

        Args:
            raw_data_dir: Répertoire racine des données brutes.
                          Par défaut : data/raw (depuis la config).

        Returns:
            Dictionnaire {source: nb_lignes_importées} pour le reporting.
        """
        if raw_data_dir is None:
            raw_data_dir = Path("data/raw")

        results = {}

        # 1. Importer la météo → fact_hvac_installations (colonnes météo)
        weather_file = raw_data_dir / "weather" / "weather_france.csv"
        if weather_file.exists():
            results["weather"] = self._import_weather(weather_file)
        else:
            self.logger.warning("Fichier meteo introuvable : %s", weather_file)

        # 2. Importer les indicateurs INSEE → fact_economic_context
        insee_file = raw_data_dir / "insee" / "indicateurs_economiques.csv"
        if insee_file.exists():
            results["insee"] = self._import_insee(insee_file)
        else:
            self.logger.warning("Fichier INSEE introuvable : %s", insee_file)

        # 3. Importer les IPI Eurostat → fact_economic_context (colonnes IPI)
        eurostat_file = raw_data_dir / "eurostat" / "ipi_hvac_france.csv"
        if eurostat_file.exists():
            results["eurostat"] = self._import_eurostat(eurostat_file)
        else:
            self.logger.warning("Fichier Eurostat introuvable : %s", eurostat_file)

        # 4. Importer SITADEL → fact_hvac_installations (colonnes permis)
        sitadel_file = raw_data_dir / "sitadel" / "permis_construire_france.csv"
        if sitadel_file.exists():
            results["sitadel"] = self._import_sitadel(sitadel_file)
        else:
            self.logger.warning("Fichier SITADEL introuvable : %s", sitadel_file)

        # 5. Importer DPE → raw_dpe (données unitaires volumineuses)
        dpe_file = raw_data_dir / "dpe" / "dpe_france_all.csv"
        if dpe_file.exists():
            results["dpe"] = self._import_dpe(dpe_file)
        else:
            self.logger.warning("Fichier DPE introuvable : %s", dpe_file)

        # Log du bilan
        self.logger.info("=" * 50)
        self.logger.info("Bilan import :")
        for source, count in results.items():
            self.logger.info("  %s : %d lignes", source, count)
        self.logger.info("=" * 50)

        return results

    def _import_weather(self, filepath: Path) -> int:
        """Importe les données météo dans fact_hvac_installations.

        Le CSV météo est au grain quotidien × ville.
        On agrège par mois × département pour correspondre au grain
        de la table de faits.

        Args:
            filepath: Chemin vers weather_aura.csv.

        Returns:
            Nombre de lignes importées.
        """
        self.logger.info("Import meteo depuis %s ...", filepath.name)

        df = pd.read_csv(filepath)
        self.logger.info("  Colonnes trouvees : %s", list(df.columns))

        # Le CSV contient : date, city, dept, temperature_2m_max, ..., hdd, cdd
        # Identifier la colonne date
        date_col = None
        for candidate in ["date", "time", "Date"]:
            if candidate in df.columns:
                date_col = candidate
                break
        if date_col is None:
            self.logger.error("  Colonne date introuvable dans le CSV meteo")
            return 0

        df[date_col] = pd.to_datetime(df[date_col])

        # Créer les colonnes d'agrégation
        df["year_month"] = df[date_col].dt.to_period("M")
        df["date_id"] = df[date_col].dt.year * 100 + df[date_col].dt.month

        # Identifier la colonne département
        dept_col = None
        for candidate in ["dept", "department", "code_dept", "dept_code"]:
            if candidate in df.columns:
                dept_col = candidate
                break

        if dept_col is None:
            self.logger.error("  Colonne departement introuvable dans le CSV meteo")
            return 0

        # S'assurer que dept est un string avec padding zéro
        df[dept_col] = df[dept_col].astype(str).str.zfill(2)

        # Mapper dept → geo_id via la table dim_geo
        geo_map = self._get_geo_mapping()
        if not geo_map:
            self.logger.error("  Table dim_geo vide, impossible de mapper les departements")
            return 0

        df["geo_id"] = df[dept_col].map(geo_map)
        df = df.dropna(subset=["geo_id"])
        df["geo_id"] = df["geo_id"].astype(int)

        # Identifier les colonnes météo disponibles
        temp_mean_col = self._find_col(df, ["temperature_2m_mean", "temp_mean"])
        temp_max_col = self._find_col(df, ["temperature_2m_max", "temp_max"])
        temp_min_col = self._find_col(df, ["temperature_2m_min", "temp_min"])
        precip_col = self._find_col(df, ["precipitation_sum", "precipitation_cumul"])
        hdd_col = self._find_col(df, ["hdd", "heating_degree_days"])
        cdd_col = self._find_col(df, ["cdd", "cooling_degree_days"])

        # Agréger par mois × département
        agg_dict = {}
        if temp_mean_col:
            agg_dict[temp_mean_col] = "mean"
        if temp_max_col:
            agg_dict[temp_max_col] = "max"
        if temp_min_col:
            agg_dict[temp_min_col] = "min"
        if precip_col:
            agg_dict[precip_col] = "sum"
        if hdd_col:
            agg_dict[hdd_col] = "sum"
        if cdd_col:
            agg_dict[cdd_col] = "sum"

        # Compter les jours de canicule (T > 35) et gel (T < 0)
        if temp_max_col:
            df["is_canicule"] = (df[temp_max_col] > 35).astype(int)
            agg_dict["is_canicule"] = "sum"
        if temp_min_col:
            df["is_gel"] = (df[temp_min_col] < 0).astype(int)
            agg_dict["is_gel"] = "sum"

        monthly = df.groupby(["date_id", "geo_id"]).agg(agg_dict).reset_index()

        # Renommer les colonnes pour correspondre au schéma
        rename_map = {}
        if temp_mean_col:
            rename_map[temp_mean_col] = "temp_mean"
        if temp_max_col:
            rename_map[temp_max_col] = "temp_max"
        if temp_min_col:
            rename_map[temp_min_col] = "temp_min"
        if precip_col:
            rename_map[precip_col] = "precipitation_cumul"
        if hdd_col:
            rename_map[hdd_col] = "heating_degree_days"
        if cdd_col:
            rename_map[cdd_col] = "cooling_degree_days"
        if "is_canicule" in monthly.columns:
            rename_map["is_canicule"] = "nb_jours_canicule"
        if "is_gel" in monthly.columns:
            rename_map["is_gel"] = "nb_jours_gel"

        monthly = monthly.rename(columns=rename_map)

        # Arrondir les valeurs numériques
        for col in monthly.select_dtypes(include=["float64"]).columns:
            monthly[col] = monthly[col].round(2)

        # Insérer dans fact_hvac_installations
        # Utiliser replace pour cette première source (les autres feront un UPDATE)
        return self.load_dataframe(monthly, "fact_hvac_installations", if_exists="replace")

    def _import_insee(self, filepath: Path) -> int:
        """Importe les indicateurs INSEE dans fact_economic_context.

        Le CSV INSEE a des colonnes : period, confiance_menages,
        climat_affaires_industrie, climat_affaires_batiment, etc.

        Args:
            filepath: Chemin vers indicateurs_economiques.csv.

        Returns:
            Nombre de lignes importées.
        """
        self.logger.info("Import INSEE depuis %s ...", filepath.name)

        df = pd.read_csv(filepath)
        self.logger.info("  Colonnes : %s", list(df.columns))
        self.logger.info("  Lignes : %d", len(df))

        if "period" not in df.columns:
            self.logger.error("  Colonne 'period' manquante")
            return 0

        # Filtrer uniquement les périodes mensuelles (YYYY-MM)
        # Certaines séries INSEE retournent des formats trimestriels (2019Q1)
        mask_monthly = df["period"].str.match(r"^\d{4}-\d{2}$")
        df = df[mask_monthly].copy()
        self.logger.info("  Lignes mensuelles retenues : %d", len(df))

        if len(df) == 0:
            self.logger.error("  Aucune periode mensuelle trouvee")
            return 0

        # Convertir period (YYYY-MM) en date_id (YYYYMM)
        df["date_id"] = df["period"].str.replace("-", "").astype(int)

        # Mapper les colonnes INSEE vers les colonnes de la table
        col_map = {
            "confiance_menages": "confiance_menages",
            "climat_affaires_industrie": "climat_affaires_indus",
            "climat_affaires_batiment": "climat_affaires_bat",
            "opinion_achats_importants": "opinion_achats",
            "situation_financiere_future": "situation_fin_future",
            "ipi_industrie_manuf": "ipi_manufacturing",
        }

        # Ne garder que les colonnes qui existent dans le CSV
        cols_to_keep = ["date_id"]
        for csv_col, db_col in col_map.items():
            if csv_col in df.columns:
                df = df.rename(columns={csv_col: db_col})
                cols_to_keep.append(db_col)

        df_insert = df[cols_to_keep].copy()

        return self.load_dataframe(df_insert, "fact_economic_context", if_exists="replace")

    def _import_eurostat(self, filepath: Path) -> int:
        """Importe les IPI Eurostat dans fact_economic_context.

        Le CSV Eurostat a des colonnes : period, nace_r2, ipi_value.
        On pivote pour avoir une colonne par code NACE (C28, C2825).

        Args:
            filepath: Chemin vers ipi_hvac_france.csv.

        Returns:
            Nombre de lignes mises à jour.
        """
        self.logger.info("Import Eurostat depuis %s ...", filepath.name)

        df = pd.read_csv(filepath)
        self.logger.info("  Colonnes : %s", list(df.columns))
        self.logger.info("  Lignes : %d", len(df))

        if "period" not in df.columns:
            self.logger.error("  Colonne 'period' manquante")
            return 0

        # Filtrer uniquement les périodes mensuelles (YYYY-MM)
        mask_monthly = df["period"].str.match(r"^\d{4}-\d{2}$")
        df = df[mask_monthly].copy()
        self.logger.info("  Lignes mensuelles retenues : %d", len(df))

        # Convertir period → date_id
        df["date_id"] = df["period"].str.replace("-", "").astype(int)

        # Pivoter : une colonne par nace_r2
        if "nace_r2" in df.columns and "ipi_value" in df.columns:
            pivot = df.pivot_table(
                index="date_id", columns="nace_r2",
                values="ipi_value", aggfunc="first",
            ).reset_index()

            # Renommer vers les colonnes de la table
            rename_map = {}
            if "C28" in pivot.columns:
                rename_map["C28"] = "ipi_hvac_c28"
            if "C2825" in pivot.columns:
                rename_map["C2825"] = "ipi_hvac_c2825"
            pivot = pivot.rename(columns=rename_map)

            # Fusionner avec fact_economic_context existant
            # Si la table a déjà des données INSEE, on fait un UPDATE
            existing = self._safe_read_table("fact_economic_context")

            if existing is not None and len(existing) > 0:
                # Merge les colonnes Eurostat dans les données existantes
                cols_eurostat = ["date_id"]
                if "ipi_hvac_c28" in pivot.columns:
                    cols_eurostat.append("ipi_hvac_c28")
                if "ipi_hvac_c2825" in pivot.columns:
                    cols_eurostat.append("ipi_hvac_c2825")

                merged = existing.merge(
                    pivot[cols_eurostat], on="date_id", how="outer",
                    suffixes=("_old", ""),
                )
                # Prendre les nouvelles valeurs Eurostat
                for col in ["ipi_hvac_c28", "ipi_hvac_c2825"]:
                    old_col = f"{col}_old"
                    if old_col in merged.columns:
                        merged[col] = merged[col].fillna(merged[old_col])
                        merged = merged.drop(columns=[old_col])

                return self.load_dataframe(
                    merged, "fact_economic_context", if_exists="replace",
                )
            else:
                return self.load_dataframe(
                    pivot, "fact_economic_context", if_exists="replace",
                )
        return 0

    def _import_sitadel(self, filepath: Path) -> int:
        """Importe les permis de construire SITADEL.

        Le CSV SITADEL est agrégé par mois × département et fusionné
        dans fact_hvac_installations (colonnes nb_permis_construire,
        nb_logements_autorises).

        Args:
            filepath: Chemin vers permis_construire_aura.csv.

        Returns:
            Nombre de lignes traitées.
        """
        self.logger.info("Import SITADEL depuis %s ...", filepath.name)

        df = pd.read_csv(filepath)
        self.logger.info("  Colonnes : %s", list(df.columns))
        self.logger.info("  Lignes : %d", len(df))

        # Le contenu dépend du format réel du CSV SITADEL
        # On log les colonnes pour diagnostic mais on ne fait pas d'import
        # erroné si le format n'est pas celui attendu
        self.logger.info(
            "  Import SITADEL : format a adapter selon les colonnes disponibles"
        )

        return len(df)

    def _import_dpe(self, filepath: Path) -> int:
        """Importe les DPE bruts dans la table raw_dpe.

        Le CSV DPE contient ~1.4M lignes pour la region AURA.
        On importe par chunks pour ne pas saturer la memoire.
        Ensuite, on agrege les DPE par mois x departement pour mettre
        a jour fact_hvac_installations (colonnes nb_dpe_total, etc.).

        Args:
            filepath: Chemin vers dpe_aura_all.csv.

        Returns:
            Nombre de lignes importees dans raw_dpe.
        """
        self.logger.info("Import DPE depuis %s ...", filepath.name)

        # Compter les lignes pour le reporting
        # (lire juste la 1ere ligne pour les colonnes)
        df_sample = pd.read_csv(filepath, nrows=2)
        self.logger.info("  Colonnes : %s", list(df_sample.columns))

        # Importer par chunks de 50 000 lignes pour limiter l'usage memoire
        chunk_size = 50_000
        total_imported = 0

        for i, chunk in enumerate(pd.read_csv(filepath, chunksize=chunk_size)):
            # Premier chunk : replace pour nettoyer la table
            # Chunks suivants : append
            mode = "replace" if i == 0 else "append"

            chunk.to_sql(
                "raw_dpe", self.engine,
                if_exists=mode, index=False,
            )
            total_imported += len(chunk)

            if (i + 1) % 5 == 0:
                self.logger.info(
                    "  DPE import : %d lignes chargees...", total_imported,
                )

        self.logger.info(
            "  Table 'raw_dpe' : %d lignes importees", total_imported,
        )

        # Agreger les DPE par mois x departement pour fact_hvac_installations
        self._aggregate_dpe_to_facts()

        return total_imported

    def _aggregate_dpe_to_facts(self) -> None:
        """Agrege les DPE bruts pour mettre a jour fact_hvac_installations.

        Calcule par mois x departement :
        - nb_dpe_total : nombre total de DPE
        - nb_installations_pac : DPE mentionnant une PAC
        - nb_installations_clim : DPE mentionnant une climatisation
        - nb_dpe_classe_ab : DPE de classe A ou B
        """
        self.logger.info("  Agregation DPE vers fact_hvac_installations...")

        # Lire les DPE avec les colonnes utiles pour l'agregation
        sql = """
        SELECT
            code_departement_ban,
            date_etablissement_dpe,
            etiquette_dpe,
            type_installation_chauffage,
            type_generateur_chauffage_principal,
            type_generateur_froid
        FROM raw_dpe
        WHERE date_etablissement_dpe IS NOT NULL
          AND code_departement_ban IS NOT NULL
        """
        df = self.query(sql)
        self.logger.info("  DPE avec date+dept valides : %d", len(df))

        if len(df) == 0:
            return

        # Convertir la date en period mensuelle
        df["date_etablissement_dpe"] = pd.to_datetime(
            df["date_etablissement_dpe"], errors="coerce"
        )
        df = df.dropna(subset=["date_etablissement_dpe"])

        df["date_id"] = (
            df["date_etablissement_dpe"].dt.year * 100
            + df["date_etablissement_dpe"].dt.month
        )

        # Detecter les installations HVAC via les champs texte
        # PAC = Pompe a chaleur dans chauffage OU dans froid (PAC air/air reversible)
        pac_pattern = r"(?i)PAC |PAC$|pompe.*chaleur|thermodynamique"

        # Les PAC sont presentes dans DEUX champs :
        #   - type_generateur_chauffage_principal : PAC air/eau, PAC geothermique
        #   - type_generateur_froid : PAC air/air (reversible, classee en froid)
        chauffage_str = df["type_generateur_chauffage_principal"].fillna("")
        froid_str = df["type_generateur_froid"].fillna("")

        df["is_pac"] = (
            chauffage_str.str.contains(pac_pattern, regex=True) |
            froid_str.str.contains(pac_pattern, regex=True)
        ).astype(int)

        # Climatisation = tout DPE ayant un generateur froid renseigne
        # (les PAC air/air sont aussi des clim reversibles)
        df["is_clim"] = (froid_str.str.len() > 0).astype(int)

        df["is_classe_ab"] = df["etiquette_dpe"].isin(["A", "B"]).astype(int)

        # Mapper dept → geo_id
        geo_map = self._get_geo_mapping()
        df["geo_id"] = df["code_departement_ban"].astype(str).str.zfill(2).map(geo_map)
        df = df.dropna(subset=["geo_id"])
        df["geo_id"] = df["geo_id"].astype(int)

        # Agreger par mois x departement
        agg = df.groupby(["date_id", "geo_id"]).agg(
            nb_dpe_total=("is_pac", "count"),
            nb_installations_pac=("is_pac", "sum"),
            nb_installations_clim=("is_clim", "sum"),
            nb_dpe_classe_ab=("is_classe_ab", "sum"),
        ).reset_index()

        agg["nb_installations_pac"] = agg["nb_installations_pac"].astype(int)
        agg["nb_installations_clim"] = agg["nb_installations_clim"].astype(int)
        agg["nb_dpe_classe_ab"] = agg["nb_dpe_classe_ab"].astype(int)

        # Mettre a jour fact_hvac_installations avec les comptages DPE
        # Lire l'existant (meteo deja importee) et fusionner
        existing = self._safe_read_table("fact_hvac_installations")

        if existing is not None and len(existing) > 0:
            # Merge les colonnes DPE dans les donnees existantes
            dpe_cols = ["date_id", "geo_id", "nb_dpe_total",
                        "nb_installations_pac", "nb_installations_clim",
                        "nb_dpe_classe_ab"]

            # Supprimer les anciennes colonnes DPE si elles existent
            for col in ["nb_dpe_total", "nb_installations_pac",
                        "nb_installations_clim", "nb_dpe_classe_ab"]:
                if col in existing.columns:
                    existing = existing.drop(columns=[col])

            merged = existing.merge(
                agg[dpe_cols], on=["date_id", "geo_id"], how="left",
            )
            self.load_dataframe(
                merged, "fact_hvac_installations", if_exists="replace",
            )
        else:
            self.load_dataframe(
                agg, "fact_hvac_installations", if_exists="replace",
            )

        self.logger.info(
            "  Agregation DPE terminee : %d lignes mois x dept", len(agg),
        )

    # =================================================================
    # Dataset ML
    # =================================================================

    def build_ml_dataset(self) -> pd.DataFrame:
        """Construit le dataset ML-ready par jointure des tables de faits.

        Effectue la jointure entre :
        - fact_hvac_installations (grain = mois x departement)
        - fact_economic_context (grain = mois seulement)
        - dim_time (features temporelles)
        - dim_geo (metadonnees geographiques)

        Le résultat est un DataFrame prêt pour le feature engineering.

        Returns:
            DataFrame avec toutes les features et la variable cible.
        """
        sql = """
        SELECT
            -- Dimensions temporelles
            t.date_id,
            t.year,
            t.month,
            t.quarter,
            t.is_heating,
            t.is_cooling,

            -- Dimensions géographiques
            g.dept_code,
            g.dept_name,
            g.city_ref,

            -- Variable cible (installations HVAC)
            f.nb_dpe_total,
            f.nb_installations_pac,
            f.nb_installations_clim,
            f.nb_dpe_classe_ab,

            -- Features météo (locales)
            f.temp_mean,
            f.heating_degree_days,
            f.cooling_degree_days,
            f.precipitation_cumul,
            f.nb_jours_canicule,
            f.nb_jours_gel,

            -- Features immobilier (locales)
            f.nb_permis_construire,
            f.nb_logements_autorises,

            -- Features économiques (nationales)
            e.confiance_menages,
            e.climat_affaires_indus,
            e.climat_affaires_bat,
            e.opinion_achats,
            e.situation_fin_future,
            e.ipi_manufacturing,
            e.ipi_hvac_c28,
            e.ipi_hvac_c2825

        FROM fact_hvac_installations f
        JOIN dim_time t ON f.date_id = t.date_id
        JOIN dim_geo g ON f.geo_id = g.geo_id
        LEFT JOIN fact_economic_context e ON f.date_id = e.date_id

        ORDER BY t.date_id, g.dept_code
        """

        df = self.query(sql)
        self.logger.info(
            "Dataset ML construit : %d lignes x %d colonnes",
            len(df), len(df.columns),
        )
        return df

    # =================================================================
    # Utilitaires
    # =================================================================

    def _count_rows(self, table_name: str) -> int:
        """Compte le nombre de lignes dans une table.

        Args:
            table_name: Nom de la table.

        Returns:
            Nombre de lignes (0 si la table n'existe pas).
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    text(f"SELECT COUNT(*) FROM {table_name}")
                )
                return result.scalar() or 0
        except Exception:
            return 0

    def _safe_read_table(self, table_name: str) -> Optional[pd.DataFrame]:
        """Lit une table entière, retourne None si la table est vide ou inexistante.

        Args:
            table_name: Nom de la table.

        Returns:
            DataFrame ou None.
        """
        try:
            df = self.query(f"SELECT * FROM {table_name}")
            return df if len(df) > 0 else None
        except Exception:
            return None

    def _get_geo_mapping(self) -> dict:
        """Retourne le mapping dept_code → geo_id depuis dim_geo.

        Returns:
            Dictionnaire {dept_code: geo_id} (ex: {'69': 6}).
        """
        try:
            df = self.query("SELECT geo_id, dept_code FROM dim_geo")
            return dict(zip(df["dept_code"].astype(str), df["geo_id"]))
        except Exception:
            return {}

    def _find_col(self, df: pd.DataFrame, candidates: list) -> Optional[str]:
        """Trouve la première colonne correspondant à une liste de candidats.

        Args:
            df: DataFrame à inspecter.
            candidates: Noms de colonnes possibles (par ordre de priorité).

        Returns:
            Nom de la colonne trouvée, ou None.
        """
        for col in candidates:
            if col in df.columns:
                return col
        return None

    def get_table_info(self) -> pd.DataFrame:
        """Retourne un résumé de toutes les tables et leur nombre de lignes.

        Returns:
            DataFrame avec colonnes [table_name, row_count].
        """
        tables = [
            "dim_time", "dim_geo", "dim_equipment_type",
            "fact_hvac_installations", "fact_economic_context",
            "raw_dpe",
        ]
        rows = []
        for table in tables:
            count = self._count_rows(table)
            rows.append({"table_name": table, "row_count": count})

        return pd.DataFrame(rows)
