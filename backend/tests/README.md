# Focused checks

`test_c04_c06_recommenders.py` contains hand-computable metric checks and
focused popularity, Pearson CF, missing-evidence, content-feature, and fallback
cases. Run it from the project root with:

```powershell
python -m pytest backend/tests/test_c04_c06_recommenders.py -q
```

The C02/C03 preparation tests remain beside their scripts in
`backend/data_scripts/`.
