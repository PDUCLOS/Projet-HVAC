# Donnees d'exemple

Ce dossier contient un extrait de 200 lignes de chaque source.
Utiliser pour tester le pipeline sans telecharger les donnees completes.

```bash
# Copier les exemples dans data/raw/ pour tester
cp -r data/sample/* data/raw/
python -m src.pipeline process
```
