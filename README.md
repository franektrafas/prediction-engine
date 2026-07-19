# Football AI

Modularny projekt AI do prognozowania wynikow meczow pilkarskich reprezentacji narodowych oraz statystyk meczowych. Projekt jest przygotowany pod Python 3.12, PostgreSQL, Google Colab oraz GitHub.

## Zakres predykcji

System obsluguje lub przygotowuje pipeline pod:

- prawdopodobienstwo `1X2`
- dokladny wynik
- gole gospodarzy i gosci
- rynki `Over/Under` 1.5, 2.5 i 3.5
- `BTTS`
- rzuty rozne
- kartki
- strzaly i strzaly celne
- `xG`
- posiadanie pilki

## Architektura

Projekt jest podzielony na niezalezne moduly:

- `football_ai/config` - konfiguracja aplikacji i aliasy nazw druzyn
- `football_ai/database` - schemat SQL, ORM SQLAlchemy, sesje i repozytoria
- `football_ai/scrapers` - konektory do StatsBomb, Football-Data, Elo, Open-Meteo, FBref i Kaggle
- `football_ai/data` - ladowanie danych, czyszczenie i normalizacja
- `football_ai/features` - generowanie cech meczowych, druzynowych i H2H
- `football_ai/models` - osobne implementacje modeli oraz ensemble
- `football_ai/training` - walidacja czasowa, tuning Optuna i orkiestracja treningu
- `football_ai/evaluation` - metryki, wykresy, SHAP i raporty
- `football_ai/predictions` - serwis inferencyjny i schematy odpowiedzi
- `football_ai/api` - REST API w FastAPI
- `football_ai/notebooks` - notebooki analityczne
- `football_ai/tests` - testy jednostkowe i integracyjne

## Repozytorium GitHub

Docelowe repozytorium projektu:

- [franektrafas/betclic.v1](https://github.com/franektrafas/betclic.v1.git)

Lokalne repo zostalo juz przygotowane pod branch `main` i zdalny `origin`.

## Tryb pracy: lokalnie + Colab

Projekt jest przygotowany pod workflow hybrydowy:

- lokalnie: edycja kodu, lekkie testy, API, czyszczenie danych, przeglad wynikow
- Google Colab: feature engineering na wiekszych zbiorach, tuning Optuna, trening LightGBM, CatBoost, XGBoost i ensemble

Jesli komputer lokalny jest slabszy, nie trzeba na nim trenowac modeli. Wystarczy:

- rozwijac kod lokalnie
- commitowac i pushowac zmiany do GitHub
- uruchamiac notebooki treningowe w Colabie

## Szybki start lokalny

Do lekkiej pracy lokalnej uzyj minimalnych zaleznosci:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-local.txt
```

To wystarcza do:

- pracy z kodem
- uruchamiania API
- lekkich testow
- przygotowania danych i struktury projektu

## Szybki start Colab

Do treningu i tuningu uzyj pelnego zestawu zaleznosci:

```bash
pip install -r requirements-colab.txt
```

W repo znajduje sie notebook startowy:

- `football_ai/notebooks/00_colab_bootstrap.ipynb`

Notebook:

- montuje Google Drive
- klonuje lub aktualizuje repo
- instaluje zaleznosci treningowe
- przygotowuje plik `.env` pod artefakty modeli na Google Drive

Poniewaz repo jest prywatne, bootstrap Colaba zaklada logowanie tokenem GitHub tylko do odczytu repozytorium.

Najlepsza opcja:

- utworz Fine-grained Personal Access Token
- nadaj mu dostep `Contents: Read-only`
- ogranicz go tylko do repo `franektrafas/betclic.v1`
- nie wklejaj tokenu do czatu ani do plikow repo

## Pelny setup developerski

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

2. Skopiuj konfiguracje:

```bash
cp .env.example .env
```

3. Uruchom PostgreSQL i zaloz baze:

```bash
createdb football_ai
```

4. Utworz tabele:

```bash
python -m football_ai.database.session
```

5. Pobierz dane:

```bash
python -m football_ai.scrapers.sources.statsbomb
python -m football_ai.scrapers.sources.football_data
python -m football_ai.scrapers.sources.elo
python -m football_ai.scrapers.sources.open_meteo
python -m football_ai.scrapers.sources.fbref
```

6. Zbuduj cechy i uruchom trening:

```bash
python -m football_ai.training.orchestrator
```

7. Uruchom API:

```bash
uvicorn football_ai.api.main:app --reload
```

## Wsparcie dla Google Colab

Kod zostal przygotowany modulowo, dzieki czemu notebooki w `football_ai/notebooks` moga importowac logike bez kopiowania kodu. W Colabie nalezy:

- uruchomic `football_ai/notebooks/00_colab_bootstrap.ipynb`
- zainstalowac zaleznosci z `requirements-colab.txt`
- zamontowac repozytorium z GitHub
- ustawic zmienne srodowiskowe z `.env`
- skierowac `MODEL_ARTIFACT_DIR` na Google Drive

Jesli repo na GitHub jest prywatne, Colab bedzie potrzebowal tokenu GitHub. Notebook `00_colab_bootstrap.ipynb` prosi o token przez ukryty prompt i nie zapisuje go do repozytorium.

## Baza danych

Schemat relacyjny znajduje sie w `football_ai/database/sql/schema.sql`, a mapowanie ORM w `football_ai/database/models.py`.

Kluczowe tabele:

- `matches`
- `teams`
- `players`
- `lineups`
- `team_statistics`
- `player_statistics`
- `elo_ratings`
- `injuries`
- `weather`
- `predictions`
- `model_versions`

## Zrodla danych

- StatsBomb Open Data
- Football-Data.co.uk
- Football API (API-Sports)
- World Football Elo Ratings
- Open-Meteo
- FBref
- pliki od uzytkownika z Kaggle

## Test Football API

Po uzupelnieniu `.env` o `FOOTBALL_API_KEY` mozna sprawdzic integracje live:

```bash
python3 -m football_ai.scrapers.sources.football_api --date 2026-07-10 --next-count 10
```

Przyklad dla konkretnej druzyny lub rozgrywek:

```bash
python3 -m football_ai.scrapers.sources.football_api --date 2026-07-10 --team-id 1095
python3 -m football_ai.scrapers.sources.football_api --date 2026-07-10 --league-id 4 --season 2026
```

Wynik zostanie:

- wyswietlony w terminalu
- zapisany jako raw JSON w `football_ai/data/raw/football_api`
- znormalizowany do kolumn zgodnych z naszym pipeline

## Trening modeli

Za trening odpowiada zestaw modeli:

- Logistic Regression
- Random Forest
- LightGBM
- CatBoost
- XGBoost
- Stacking Ensemble

Kazdy model posiada:

- osobny plik
- osobny pipeline
- zapis `joblib`
- wczytywanie
- predykcje

## Walidacja i ewaluacja

Projekt stosuje:

- podzial chronologiczny `train/validation/test`
- `TimeSeriesSplit` dla walidacji podczas tuningu
- metryki klasyfikacyjne i regresyjne
- kalibracje, ROC, PR, confusion matrix
- feature importance i SHAP

## Testy

```bash
python -m unittest discover football_ai/tests
```

## CI

Workflow GitHub Actions uruchamia:

- instalacje zaleznosci
- testy jednostkowe
- podstawowa weryfikacje importow
- harmonogram odswiezania danych do uruchamiania po skonfigurowaniu sekretow i danych wejsciowych

## Co bedzie potrzebne pozniej od Ciebie

Na obecnym etapie nie musisz jeszcze nic dosylac poza repo. W kolejnych krokach przydadza sie:

- pobrane pliki z Kaggle
- eksporty FBref, jesli chcesz ich uzywac jako lokalnego wsadu
- decyzja, gdzie na Google Drive mamy trzymac modele, dane i raporty
- dane dostepowe do PostgreSQL, jesli baza ma dzialac poza lokalnym srodowiskiem i CI

## Dalszy rozwoj

Architektura zostala przygotowana tak, aby mozna bylo:

- dodawac nowe zrodla danych bez przepisywania pipeline'u
- trenowac wiele targetow w ramach wspolnego rejestru modeli
- podpinac nowsze modele i dodatkowe strategie ensemble
- rozszerzac API o batch scoring i wersjonowanie predykcji
