# Prompt do Claude Code — International Football Data Engine

> To jest duże, wielofazowe zadanie architektoniczne, nie jednorazowy skrypt.
> Nie próbuj zrobić wszystkiego w jednej sesji — trzymaj się kolejności
> faz na końcu. Specyfikacja poniżej pochodzi wprost od użytkownika
> (przemyślana, nie skracaj jej) plus dopisany przeze mnie załącznik ze
> zweryfikowanymi już źródłami (sekcja "Co już sprawdziłem"), żeby nie
> zaczynać researchu od zera. Skopiuj wszystko poniżej linii.

---

## Kontekst i cel

Najważniejszym obecnym problemem projektu nie jest już wybór kolejnych
modeli ML, lecz zbudowanie legalnego, wysokiej jakości i automatycznie
aktualizowalnego źródła danych o meczach reprezentacji narodowych.

Projekt ma docelowo być produktem komercyjnym, dlatego dane używane do
treningu modeli muszą mieć jasno zweryfikowane prawa do użycia w Machine
Learning, a w przyszłości także do zastosowań komercyjnych.

Buduj moduł **International Football Data Engine**, który automatycznie
zbiera, normalizuje, waliduje, wersjonuje i przechowuje dane o meczach
reprezentacji:

```text
                  DATA SOURCES
                       │
        ┌──────────────┼───────────────┐
        │              │               │
      Results       Match Stats      Squads
        │              │               │
        └──────────────┼───────────────┘
                       ↓
                  Normalization
                       ↓
                 Data Validation
                       ↓
                License Validation
                       ↓
              Data Provenance Layer
                       ↓
                  PostgreSQL
                       ↓
              Feature Engineering
                       ↓
                 ML / Analytics
```

Nie szukamy jednej idealnej bazy — szukamy najlepszej kombinacji
**jakości + pokrycia + legalności + reprodukowalności**, nie "największej
liczby danych".

## Najważniejsza zasada prawna

Nie zakładaj, że publicznie dostępna strona internetowa może być
automatycznie scrapowana i używana do trenowania modeli.

Nie używaj do produkcyjnego ML danych, jeżeli:

- regulamin źródła zabrania scrapowania,
- regulamin zabrania użycia danych do ML/AI,
- licencja jest nieznana,
- licencja nie zezwala na użycie komercyjne,
- nie można ustalić pochodzenia danych,
- dataset deklaruje np. CC0, ale zawiera dane pochodzące z zewnętrznego
  źródła, którego prawa nie są jasne.

Jeżeli licencja lub prawa są niejasne: **`DO NOT USE FOR PRODUCTION ML`**.
Można przechowywać takie źródło w oddzielnej warstwie
`research`/benchmark, ale nie może zasilać produkcyjnego datasetu
treningowego.

**Rozdziel zawsze `LICENSE OF CODE` od `RIGHTS TO DATA`.** Licencja MIT na
kod repozytorium (scraper, importer) nie oznacza automatycznie żadnej
licencji na dane, które ten kod pobiera z zewnętrznego źródła (np.
repozytoria pobierające z publicznych endpointów FIFA).

---

## Co już sprawdziłem (20 sierpnia 2026) — nie zaczynaj od zera

Poniższe zweryfikowałem realnie (fetch/przeglądarka), nie z pamięci. Krok
1 poniżej ("znajdź 20-30 datasetów") ma już te ~8 jako punkt startowy.

| Źródło | Status | Uwagi |
|---|---|---|
| `martj42/international_results` (GitHub) | **PRODUCTION_ML** | CC0, już zintegrowane w `scrapers/sources/international_results.py`. Wyniki + strzelcy + rzuty karne, 1872-2026. |
| `footballcsv/cache.footballsquads` (GitHub) | **PRODUCTION_ML** (składy) | CC0, cache HTML→CSV footballsquads.co.uk, folder `national/` = worldcup/eurocham/copaamer/africacn/confed. Plan w `PROMPT_FOOTBALLSQUADS.md` (jeszcze niezaimplementowany). |
| `openfootball/worldcup.json` (+ siostrzane repo `openfootball/euro.json` — do sprawdzenia) | **PRODUCTION_ML** | Explicite: *"schema, data and scripts are dedicated to the public domain. Use as you please with no restrictions whatsoever."* Wyniki + strzelcy (minuta, karny) + stadion/miasto. Tylko MŚ (klub i narodowe) w tym repo — organizacja `github.com/openfootball` ma prawdopodobnie osobne repo dla Euro i innych, sprawdź. |
| `jfjelstul/worldcup` — Fjelstul World Cup Database (GitHub) | **PRODUCTION_ML z zastrzeżeniem** | Czyste pochodzenie (kodowane z Wikipedii, krzyżowo weryfikowane z oficjalnymi raportami FIFA — nie scraping płatnego dostawcy). CC-BY-SA 4.0 potwierdzone: komercyjne użycie dozwolone, ale wymaga atrybucji i share-alike dla utworów pochodnych — **skonsultuj z kimś kompetentnym prawnie**, co dokładnie liczy się jako "utwór pochodny" przy użyciu jako sygnał treningowy modelu, zanim to pójdzie do produkcyjnego API. Tylko MŚ 1930-2022 (brak WC2026), mało "klasycznych" team stats (shots/corners/possession) — głównie matches/goals/bookings/substitutions/squads/players/managers/referees. **UWAGA:** nie mylić z ich hostowanym produktem WorldCups.ai — ten jest CC-BY-NC-SA (tylko niekomercyjnie). Brać dane tylko z GitHuba. |
| RSSSF (rsssf.org) | **RESEARCH_ONLY** | Realne, ogromne archiwum, ale to katalog/hub stron per kraj/rozgrywki, nie jeden ustrukturyzowany dataset. Własna stopka: *"free to copy this document in whole or part provided that proper acknowledgement is given"* — permisywne, ale to nie jest maszynowo czytelny dataset z jasną licencją per plik. Do oceny źródło-po-źródle, nie hurtowo. |
| `worldfootballR` (pakiet R) i `worldfootballR_data` (GitHub) — dane z **FBref.com** | **ZAKAZANE, nie RESEARCH_ONLY nawet** | Regulamin właściciela (Sports-Reference/FBref) wprost: *"copy or use any material or Content from the Site... for purposes of training, fine-tuning, prompting, or instructing artificial intelligence models... including... supporting machine learning methods used to predict, classify, label, or score inputs into the models"* + zakaz automatycznego dostępu bez zgody + grożą blokadą IP. To nie jest niejasność licencji — to wprost nazwany zakaz dokładnie naszego zastosowania. Nie buduj wokół tego niczego, nawet jako research-only cache. Dodatkowo oba repo są zarchiwizowane/nieutrzymywane od 09.2025. |
| Kaggle `Mominullptr/fifa-world-cup-2026-dataset` (+ mirror na Hugging Face i GitHub) | **MIESZANE — tylko część PRODUCTION_ML** | CC0 zadeklarowane przez autora, ale własny README repo wprost mówi, że `match_team_stats.csv` (possession/shots/SOT/corners/fouls/offsides/saves — czyli dokładnie to, czego szukamy) jest *"sourced from verified providers... FIFA, Sofascore, FBref, etc."* — czyli częściowo dziedziczy zakaz FBref, niezależnie od deklaracji CC0 (autor nie ma prawa relicencjonować cudzych danych). Tabelę `match_team_stats` oznacz `RESEARCH_ONLY` / nie używaj. Pozostałe tabele (`teams`, `venues`, `matches` — wynik/data, `squads_and_players` bez `market_value_eur`) wyglądają czyściej (FIFA.com), ale zweryfikuj `market_value_eur` osobno (prawdopodobnie Transfermarkt — inny regulamin do sprawdzenia) i `referees.csv`/`avg_cards_per_game` (nieznane źródło). |
| Kaggle `nalisha/fifa-world-cup-2022-match-performance-dataset` | **USUNIĘTE / martwy link** | Strona zwraca "We can't find that page" (sprawdzone dwoma URL-ami). Skreśl z listy kandydatów, nie ma czego oceniać. |

**Pozostałe kategorie do przeszukania** (użytkownik wymienił, jeszcze nie
sprawdzone): Zenodo, dodatkowe datasety Kaggle/Hugging Face poza
powyższymi, oficjalne portale open-data federacji (FIFA, UEFA, CONMEBOL,
CAF, AFC) i organizatorów turniejów, inne repo w organizacji
`github.com/openfootball` (euro.json, inne rozgrywki), inne publiczne
projekty badawcze.

---

## Przykładowe źródła do dalszej weryfikacji (z oryginalnej listy)

Dla każdego nowego kandydata zweryfikuj to samo, co powyżej: licencję,
**faktyczne** upstream pochodzenie danych (nie tylko deklarację
uploadera), możliwość użycia ML, możliwość zastosowania komercyjnego, i
czy zabrania tego regulamin źródła (nie tylko licencja pliku).

Konkretne dziury do zamknięcia:

1. **Inne wersje "FIFA World Cup Match Stats"** na Kaggle/Hugging
   Face/GitHub — sprawdź lata, kompletność, statystyki, licencję, źródło.
2. **Projekty GitHub pobierające dane z publicznych endpointów FIFA** —
   sprawdź regulamin samego FIFA.com/FIFA API dla automatycznego dostępu,
   nie tylko licencję kodu repozytorium, które go odpytuje.
3. **FootballSquads** (footballsquads.co.uk, przez cache
   `footballcsv/cache.footballsquads`) — mimo że sam cache jest CC0,
   sprawdź jeszcze regulamin oryginalnej strony pod kątem ML/komercyjnego
   użycia specyficznie (nie tylko robots.txt, który już sprawdziłem — tylko
   blokuje `/cgi-bin/` i `/phpBB2/`). Jeśli prawa do *oryginalnych* danych
   (nie do cache'u) są niejasne, oznacz jako `RESEARCH_ONLY`, nie
   `PRODUCTION_ML`, dopóki nie ma jasności.

---

## Warstwowa jakość danych

Nie wymagaj, żeby każdy historyczny mecz miał wszystkie statystyki:

```text
1988: goals ✓  cards ✓  shots ?  corners ?  xG ✗
2005: goals ✓  cards ✓  shots ✓  corners ✓  possession ✓  xG ?
2026: goals ✓  cards ✓  shots ✓  corners ✓  possession ✓  xG ✓  events ✓  lineups ✓
```

To jest prawidłowe. Nie wolno: imputować xG zera, wpisywać średniej jako
prawdziwej wartości historycznej, tworzyć fikcyjnych statystyk. Brak
danych zostaje `NULL` lub jawnie oznaczony jako missing.

## Data Provenance Layer

Każdy rekord/statystyka przechowuje pochodzenie. Minimalny schemat:

```text
match_id
statistic_name
home_value
away_value
source
source_url
source_dataset
source_license
license_status
retrieved_at
data_version
confidence
```

Przykład:

```text
match_id = 23891
statistic = shots
home_value = 14
away_value = 8
source = FIFA_WC_2026_DATASET
source_license = CC0
license_status = VERIFIED
retrieved_at = 2026-08-20
confidence = HIGH
```

## Confidence Layer

```text
HIGH     — FIFA official / open licensed dataset
MEDIUM/HIGH — niezależny dataset ze zweryfikowanym pochodzeniem
LOW      — dataset z niejasnym upstream pochodzeniem
UNKNOWN  — nieznane źródło
```

Dane `UNKNOWN` nie mogą trafiać do produkcyjnego treningu.

## Source Priority i konflikty

```text
OFFICIAL LICENSED DATA
    ↓
OPEN LICENSED DATA
    ↓
VERIFIED INDEPENDENT DATASET
    ↓
RESEARCH DATASET
    ↓
UNKNOWN
```

Jeśli ta sama statystyka istnieje w kilku źródłach z różnymi wartościami:
wykryj konflikt, zachowaj wszystkie wartości (nie usuwaj żadnego źródła),
wybierz wartość canonical wg Source Priority, zapisz informację o
konflikcie.

## Canonical Match Model

```text
Match:
  match_id, date, home_team, away_team, competition, competition_type,
  tournament, stage, venue, city, country, referee, attendance, weather,
  home_score, away_score, extra_time, penalties
```

Osobne relacyjne tabele: `match_team_stats`, `match_events`,
`match_lineups`, `match_player_stats`, `match_officials`,
`match_sources`.

**Integracja z istniejącym schematem:** ten projekt tworzy tabele przez
SQLAlchemy `Base.metadata.create_all()` (`database/session.py`), nie
uruchamiając `schema.sql` bezpośrednio — dodaj nowe modele do
`database/models.py` w tym samym stylu co istniejące (`Match`, `Team`,
`Competition` już tam są, `competition_type` = `international`/`club`
już istnieje jako konwencja nazewnictwa) i ręcznie zsynchronizuj
`database/sql/schema.sql`, tak jak reszta projektu to dziś robi (nie ma
tu jeszcze Alembic).

`match_team_stats` — pola na: shots, shots_on_target, corners, fouls,
yellow_cards, red_cards, offsides, possession, passes, pass_accuracy,
xG, saves, free_kicks, throw_ins, crosses, tackles, interceptions. Nie
zakładaj, że wszystkie muszą wystąpić w każdym rekordzie.

**Event model:** goal, own_goal, penalty, yellow_card, second_yellow,
red_card, substitution, var_event, missed_penalty. Opcjonalne pola:
minute, added_time, player, assist, team, related_player, event_source.

**Squad data** (jeśli legalne źródła pozwalają): player_id, player_name,
team, position, starter, substitute, minutes, shirt_number. Zasila
później: National Team Continuity, Experience, lineup-based features
(patrz też `PROMPT_FOOTBALLSQUADS.md` dla konkretnie tego). **Nie
implementuj** Team Chemistry zależnego od przewidywanej XI, dopóki nie ma
wiarygodnego i legalnego źródła *aktualnych* składów.

## Feature engineering — na razie NIE nowe cechy

Najpierw solidna baza: rolling goals/xG/shots/SOT/corners/cards/
possession, Elo, strength of schedule, rest days, match importance, home
advantage, H2H, momentum. Dopiero po uzyskaniu odpowiedniej jakości
danych: Team Chemistry, Squad Stability, Expected Lineup Strength, Club
Form Index, Injury Impact. Nie wdrażaj cechy, której dane wejściowe nie
są wystarczająco wiarygodne.

**Brak data leakage — nienegocjowalne, zgodnie z resztą projektu**
(patrz `features/builder.py`'s już istniejące `.shift(1)`-based cechy):
każda cecha dla meczu w dniu D liczona wyłącznie z informacji dostępnych
`<= D - kickoff_time`. Nigdy przyszłych rankingów/wyników/wartości/
składów/danych opublikowanych po meczu.

**Nie zmieniaj `features/builder.py` na podstawie tego zadania bez
osobnej, jawnej decyzji** — dokładnie ta sama zasada co przy błędzie
neutral-venue z tego tygodnia. Ten prompt buduje warstwę *pod* silnikiem
(zbieranie/walidacja/provenance), nie sam silnik.

## Research vs Production

Dwa poziomy: `research_data` (może zawierać interesujące, ale prawnie
niepewne źródła) i `production_data` (wyłącznie dane z potwierdzonym
prawem użycia). Pozwala eksperymentować bez zanieczyszczania
produkcyjnego datasetu.

## Automatyczne testy

Sprawdzaj: duplicate matches, duplicate events, impossible scores,
negative statistics, possession > 100, cards < 0, corners < 0, shots <
shots_on_target, away_team == home_team, date inconsistencies,
impossible event timestamps, konflikty między źródłami, brakujący
source, brakujący license_status.

Ten projekt trzyma testy w `football_ai/tests/unit/` (unittest,
`pytest`-kompatybilne) — dodaj nowe w tym samym stylu, z golden-testami
na znanych, ręcznie zweryfikowanych meczach (ten sam wzorzec co
`insights/` moduł: min. kilkanaście testów na prawdziwych, sprawdzalnych
faktach, nie tylko "kod się uruchamia").

## Data Quality Report

Po każdym imporcie generuj raport: total matches, matches with
shots/SOT/corners/cards/possession/xG/lineups, coverage by year/
competition/source, missingness, conflicts, unverified licenses.

Przykład:

```text
1980–1989: shots 12%  corners 18%  cards 72%
2010–2019: shots 87%  corners 91%  cards 98%
2020–2026: shots 99%  corners 99%  cards 100%  xG 94%
```

## Kolejność wykonania

1. Znajdź 20-30 potencjalnych datasetów (patrz tabela "Co już
   sprawdziłem" — to jest Twój punkt startowy, nie zaczynaj od zera).
2. Dla każdego zapisz: URL, zakres dat, zakres rozgrywek, statystyki,
   licencję, upstream source, możliwość użycia ML, możliwość
   komercyjnego wykorzystania, sposób pobrania, format, jakość.
3. Odfiltruj źródła, których nie można bezpiecznie używać.
4. Zbuduj ranking źródeł (Source Priority).
5. Zaimplementuj importery najlepszych źródeł (wzorem
   `scrapers/sources/international_results.py` — `BaseDataSource`,
   `fetch_text`/`fetch_json`, jeden moduł per źródło).
6. Zbuduj normalizację (wspólny Canonical Match Model, kanonikalizacja
   nazw drużyn przez istniejące `data.normalization.canonicalize_team_name`).
7. Zbuduj Data Provenance Layer.
8. Zbuduj Data Quality Layer (testy + raport pokrycia).
9. Zaimportuj dane do PostgreSQL.
10. Wygeneruj raport pokrycia.
11. Dopiero wtedy kontynuuj feature engineering i ML — i to też jako
    osobna, jawna decyzja, nie automatyczny follow-up.

Nie dodawaj do produkcji żadnego źródła tylko dlatego, że jest łatwe do
pobrania. Jeżeli nie da się potwierdzić praw do użycia danych, oznacz je
jako `RESEARCH_ONLY` i nie używaj do produkcyjnego ML.
