# Prompt do Claude Code — zaplanowane odświeżanie danych meczowych

> To jest realne zadanie do zaimplementowania, nie diagnostyka. Nie zgaduj
> co do dostępności sieci ani mechanizmu — zweryfikuj to w swoim własnym
> środowisku wykonania, zanim zdecydujesz, i przetestuj całość end-to-end
> zanim uznasz temat za zamknięty. Skopiuj wszystko poniżej linii.

---

## Kontekst i cel

Lokalne dane meczowe stoją w miejscu. `football_ai/data/processed/international_results_full_history.parquet`
(i pochodne: `_enriched`, `_training_history`, `_modern_with_elo`, `goalscorers`)
mają datę modyfikacji **2 sierpnia 2026**. Zero automatyzacji odświeżania —
sprawdzone: brak cron, brak GitHub Action, brak jakiegokolwiek zaplanowanego
zadania w repo. To nie jest teoretyczna luka: ASEAN Cup trwa od 24 lipca
2026, więc cokolwiek dziś stoi za Streamlitem, PDF-ami czy API `/v1/matches`,
prawdopodobnie pomija całą końcówkę fazy grupowej i fazę pucharową tego
turnieju, a każdy kolejny mecz reprezentacji od 2 sierpnia w ogóle nie
istnieje w danych.

Decyzja o statystykach zespołowych (shots/corners/cards/possession) jest
osobna i już podjęta — patrz `PLAN_SILNIKA.md`, notatka z 25 sierpnia 2026:
zostajemy przy obecnym pułapie, bez płatnego api-football.com. To zadanie
dotyczy wyłącznie **wyników meczów i podstawowych danych** (martj42/Elo/
nadchodzące terminarze) — rzeczy, które już legalnie zbieramy, tylko nie na
bieżąco.

## Ważne ustalenie z sesji Cowork — nie zakładaj, że gdziekolwiek to
## odpalisz będzie miało dostęp do sieci

Sprawdziłem to bezpośrednio z sandboksa tej sesji Cowork: brak ogólnego
dostępu do internetu poza wąską listą dozwolonych domen. Bezpośredni test
curl na `raw.githubusercontent.com` (martj42), `v3.football.api-sports.io`,
`api.football-data.org`, a nawet zwykły `google.com` — wszystkie zwróciły
odmowę połączenia (connection refused / tunnel 403). Zadanie cykliczne
uruchamiane przez mechanizm Cowork dziedziczyłoby te same ograniczenia i
codziennie próbowałoby się połączyć bez żadnego efektu — cicha, fałszywa
automatyzacja, gorsza niż brak automatyzacji, bo daje złudne poczucie, że
problem jest rozwiązany.

**Twoje środowisko wykonania może być inne** (lokalna maszyna użytkownika,
inny sandbox, CI) — ale nie zakładaj tego, tylko zweryfikuj realnym
`curl`/`requests.get` na te same domeny przed budową czegokolwiek, dokładnie
tak jak zrobiłem to tutaj.

## Zadanie

1. **Ustal mechanizm, na podstawie realnej weryfikacji, nie zgadywania:**
   - Sprawdź `git remote -v` — czy to repo faktycznie jest hostowane na
     GitHubie? Jeśli tak, GitHub Actions z triggerem `schedule: cron:` jest
     prawdopodobnie najczystszą opcją — działa niezależnie od tego, czy
     czyjkolwiek komputer jest włączony, ale wymaga trzymania
     `FOOTBALL_API_KEY` jako GitHub Secret, nie w pliku `.env`.
   - Jeśli nie ma GitHuba albo repo jest wyłącznie lokalne — lokalny
     cron/launchd (macOS) na maszynie, gdzie realnie ląduje wynik. Wypisz
     dokładną instrukcję (plik `.plist` dla launchd albo linijkę crontab)
     zamiast zakładać, że użytkownik sam to złoży.
   - Nie wybieraj mechanizmu bez sprawdzenia, że wybrane środowisko
     faktycznie ma dostęp do potrzebnych domen.

2. **Napraw przy okazji realny bug w `.env`:** linijka
   `FOOTBALL_API_KEY= <wartość>` ma spację po `=`, co psuje naiwne
   `source .env` w bashu (choć `python-dotenv`, którego projekt już używa,
   może to obsłużyć poprawnie — sprawdź czy faktycznie obsługuje, i
   ujednolić formatowanie całego pliku niezależnie od tego).

3. **Zbuduj jeden skrypt/wrapper odświeżający** (np.
   `football_ai/scrapers/refresh_all.py` albo `scripts/refresh_data.sh` —
   Twój wybór), który po kolei odpala źródła istotne dla świeżości wyników:
   - `international_results` (martj42) — główne źródło świeżości wyników
   - `elo`
   - `football_api` — terminarz/wyniki nadchodzących i bieżących meczów
     (wymaga `FOOTBALL_API_KEY`)
   - inne istniejące moduły z `football_ai/scrapers/sources/`, jeśli
     faktycznie wpływają na świeżość wyników — nie musisz wpinać
     statsbomb/fbref do codziennego zadania, jeśli nie są częścią tego
     problemu; oceń sam, nie dodawaj na wyrost.
   - Błąd pojedynczego źródła (limit, sieć) ma być logowany i pomijany, nie
     przerywać całości — ten wzorzec już istnieje w `football_api.py`,
     trzymaj się go.
   - Krótkie podsumowanie na końcu: ile wierszy dodano/zaktualizowano per
     źródło, zakres dat, znacznik czasu — zapisane gdzieś sprawdzalnym
     (wystarczy plik log).

4. **Częstotliwość:** raz dziennie wystarczy na start. Zrób to tak, żeby
   łatwo było podbić częstotliwość później (np. w trakcie żywego turnieju),
   bez przepisywania całości.

5. **Przetestuj naprawdę, end-to-end**, zanim zgłosisz gotowość: realne
   odpalenie, potwierdzone zmienione znaczniki czasu plików parquet, brak
   wyjątków, realne liczby wierszy przed/po.

## Co zgłosić z powrotem

1. Wybrany mechanizm i **dlaczego** — w tym wynik sprawdzenia, czy repo jest
   na GitHubie, i wynik testu dostępności sieci ze środowiska, w którym to
   faktycznie uruchomiłeś.
2. Potwierdzenie realnego przebiegu end-to-end: liczby wierszy przed/po per
   źródło, nowy zakres dat, brak cichych błędów.
3. Skonfigurowana częstotliwość i gdzie lądują logi/podsumowanie.
4. Czy `.env` bug został naprawiony i czy `python-dotenv` faktycznie sobie z
   nim radził, czy to była realna usterka.

## Czego nie robić

- Nie ruszaj `features/builder.py` ani `training/` w ramach tego zadania —
  to osobny, już opisany wątek (gate neutral-venue).
- Nie dodawaj nowych źródeł danych ponad te potrzebne do świeżości wyników
  — to nie jest okazja do rozszerzania zakresu.
- Nie zakładaj dostępności sieci na podstawie tego, co napisane wyżej o
  sandboksie Cowork — to dotyczy tamtego środowiska, nie automatycznie
  Twojego. Zweryfikuj własne.
