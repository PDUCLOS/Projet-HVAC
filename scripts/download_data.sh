#!/usr/bin/env bash
# =============================================================================
# HVAC Market Analysis — Telechargement des donnees
# =============================================================================
# Telecharge les donnees depuis pCloud ou les recollecte depuis les APIs.
#
# Usage :
#   ./scripts/download_data.sh              # Depuis pCloud (rapide)
#   ./scripts/download_data.sh --api        # Depuis les APIs (lent, ~1h)
#   ./scripts/download_data.sh --demo       # Donnees de demo (offline)
# =============================================================================

set -e

MODE=${1:---pcloud}

echo "============================================="
echo "  HVAC Market Analysis — Donnees"
echo "  Mode : $MODE"
echo "============================================="

case "$MODE" in
    --pcloud|-p)
        echo ""
        echo "  Telechargement depuis pCloud..."
        echo "  Lien : https://e.pcloud.link/publink/show?code=kZbQQ3Zg1slD5WfRgh42fH5rRpDDYWyBEsy"
        echo ""
        python -m src.pipeline sync_pcloud
        echo ""
        echo "  Donnees telechargees dans data/raw/"
        echo "  Lancer le traitement : python -m src.pipeline process"
        ;;

    --api|-a)
        echo ""
        echo "  Collecte depuis les APIs (necessite acces internet)..."
        echo "  Attention : la collecte DPE peut prendre ~1h"
        echo ""
        python -m src.pipeline collect
        echo ""
        echo "  Donnees collectees dans data/raw/"
        echo "  Lancer le traitement : python -m src.pipeline process"
        ;;

    --demo|-d)
        echo ""
        echo "  Generation des donnees de demonstration..."
        echo "  (donnees synthetiques realistes, pas d'internet requis)"
        echo ""
        python scripts/generate_demo_data.py
        echo ""
        echo "  Donnees generees dans data/raw/"
        echo "  Lancer le traitement : python -m src.pipeline process"
        ;;

    --sample|-s)
        echo ""
        echo "  Copie des donnees d'exemple..."
        if [ -d "data/sample" ]; then
            cp -r data/sample/* data/raw/
            echo "  Donnees copiees dans data/raw/"
        else
            echo "  ERREUR: data/sample/ non trouve."
            echo "  Generez d'abord les exemples : python scripts/create_sample_data.py"
            exit 1
        fi
        ;;

    --help|-h)
        echo ""
        echo "  Options :"
        echo "    --pcloud, -p  Telecharger depuis pCloud (rapide, defaut)"
        echo "    --api, -a     Collecter depuis les APIs (lent, ~1h)"
        echo "    --demo, -d    Generer des donnees de demo (offline)"
        echo "    --sample, -s  Copier les donnees d'exemple (200 lignes)"
        echo ""
        ;;

    *)
        echo "  Option inconnue : $MODE"
        echo "  Utilisez --help pour voir les options"
        exit 1
        ;;
esac
