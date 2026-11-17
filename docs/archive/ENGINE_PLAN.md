# Plan rozwoju silnika — pionowa ścieżka do produktu

**Data:** 29 lipca 2026
**Status:** decyzja podjęta, do wykonania

---

## Decyzja

**Wstrzymujemy:** rozszerzanie na top 5 lig i Ligę Mistrzów, dalsze dostrajanie modelu,
prace nad GNN.

**Robimy:** jedną kompletną pionową ścieżkę od surowych danych do gotowego,
opublikowalnego zdania — na rozgrywkach reprezentacyjnych, które już mamy.

**Termin naturalny:** start Ligi Narodów 24 września 2026. Polska gra, jest wartość
informacyjna, jest deadline.

---

## Zasada nadrzędna

Silnik predykcyjny jest **wystarczająco dobry**. Kolejne punkty procentowe skuteczności
nie zmieniają nic biznesowo (sufit rynkowy ~54%, darmowe kursy są lepiej skalibrowane).

Wartość produktu leży w warstwach, których dziś nie ma:

```
[ dane historyczne ]  ← luka: mamy tylko 2000+
        ↓
[ silnik predykcyjny ]  ← GOTOWE, nie ruszać
        ↓
[ symulacje ]  ← brak
        ↓
[ wykrywanie faktów + weryfikacja ]  ← brak, TO JEST RDZEŃ PRODUKTU
        ↓
[ narracja ]  ← brak
        ↓
[ API / prezentacja ]  ← częściowo
```

Każdy krok poniżej silnika jest tańszy niż ulepszanie silnika i wart wielokrotnie więcej.

---

## Kolejność wykonania

Kolejność wynika z zależności, nie z atrakcyjności.

### Krok 1 — Pełna historia (martj42)

**Dlaczego pierwszy:** blokuje wszystko inne. Rekord „pierwszy raz od 1962" jest niemożliwy
do policzenia na danych od 2000 roku.

**Co zrobić:**
- Nowy moduł `football_ai/scrapers/sources/international_results.py`
- Źródło: repozytorium `martj42/international_results` (GitHub, CSV, aktualizowane automatycznie)
- Cztery pliki: `results.csv` (~49 tys. meczów od 1872), `goalscorers.csv`,
  `shootouts.csv`, `former_names.csv`
- Scalanie z istniejącym `international_results_modern_with_elo.parquet` po kluczu
  (data, drużyna gospodarzy, drużyna gości) — z rozstrzyganiem konfliktów na korzyść
  martj42 dla wyników, na korzyść naszych danych dla xG/statystyk
- `former_names.csv` wpiąć w `config/team_aliases.py` — to rozwiązuje RFN/NRD, ZSRR,
  Jugosławię, Czechosłowację, czyli problem, którego dziś nie obsługujemy

**Decyzja techniczna do podjęcia w trakcie:**
Dane od 1872 są bezcenne dla **rekordów**, ale szkodliwe dla **treningu modelu** — piłka
z 1900 roku to inna gra i inne rozkłady. Wprowadzić rozdział:
- `records_frame` — pełna historia, bez cięcia
- `training_frame` — cięcie od ~1990 (do ustalenia empirycznie)

Nie wolno bezrefleksyjnie wrzucić 1872 roku do `build_features` i trenować na całości.

**Kryterium ukończenia:**
- `python -m football_ai.scrapers.sources.international_results` zapisuje pełny parquet
- liczba meczów > 45 000, zakres dat 1872–2026
- test sprawdzający, że mecz Niemiec z 1974 i mecz RFN z 1974 trafiają do właściwej encji

---

### Krok 2 — Moduł faktów i weryfikacji (rdzeń produktu)

**Dlaczego drugi:** to jest jedyna rzecz w całym planie, której nie da się dostać
za darmo gdzie indziej. Tu leży cała obrona produktu.

**Co zrobić:**

Nowy pakiet `football_ai/insights/`:

```
insights/
├── __init__.py
├── schemas.py       # Fact: treść, typ, waga, provenance (lista match_id)
├── streaks.py       # serie: zwycięstw, bez porażki, bez gola, u siebie/na wyjeździe
├── records.py       # rekordy: największa wygrana, najdłuższa seria, pierwszy raz od...
├── milestones.py    # kamienie milowe: setny mecz, 50. gol selekcjonera
├── h2h.py           # historia bezpośrednich starć
├── detectors.py     # reguły „czy to jest nietypowe" (odchylenie od mediany, rzadkość)
└── verification.py  # każdy fakt musi mieć provenance i dać się odtworzyć
```

**Zasada architektoniczna — nienegocjowalna:**
Każdy `Fact` niesie ze sobą listę `match_id`, z których powstał. Fakt bez provenance
nie może opuścić modułu. Powód: dziennikarz publikuje to pod swoim nazwiskiem, a błąd
rzeczowy niszczy zaufanie do całego produktu.

**Testy — inaczej niż w reszcie projektu:**
Nie wystarczy „kod się uruchamia". Potrzebne są **golden tests** na znanych, ręcznie
zweryfikowanych faktach historycznych. Przykład: „Polska nie wygrała z Niemcami do 2014" —
jeśli moduł tego nie odtworzy, jest zepsuty.

Minimum 15 takich testów przed uznaniem modułu za gotowy.

**Kryterium ukończenia:**
- dla dowolnego meczu z bazy moduł zwraca listę faktów z provenance
- wszystkie golden testy przechodzą
- `detectors.py` potrafi uszeregować fakty według nietypowości, a nie zwracać wszystkiego

---

### Krok 3 — Symulacje Monte Carlo

**Dlaczego trzeci:** najtańszy w budowie (silnik prawdopodobieństw już jest), a produkuje
treść niedostępną w żadnym darmowym źródle.

**Co zrobić:**

Nowy pakiet `football_ai/simulation/`:

```
simulation/
├── __init__.py
├── monte_carlo.py    # N symulacji pozostałych meczów
├── competition.py    # struktura rozgrywek: grupy, tabela, awans
└── scenarios.py      # „co jeśli" — modyfikacja wsadu i ponowne przeliczenie
```

Wejście: lista pozostałych meczów + prawdopodobieństwa z istniejącego
`predictions/service.py`. Wyjście: rozkład końcowych pozycji, szanse na awans,
najbardziej prawdopodobne ścieżki.

**Uwaga na pułapkę:** symulacja musi losować **wyniki**, nie tylko 1X2 — bo o awansie
decyduje bilans bramek. Losowanie z rozkładu Poissona na podstawie przewidywanych goli,
nie z trzech kategorii.

**Zależność zewnętrzna:** terminarz Ligi Narodów — do pobrania przez istniejący
`scrapers/sources/football_api.py`.

**Kryterium ukończenia:**
- „szanse Polski na awans z grupy" liczone w mniej niż 5 sekund dla 10 000 symulacji
- scenariusz „co jeśli" zmienia wynik w oczekiwanym kierunku
- suma prawdopodobieństw awansu w grupie zgadza się z liczbą miejsc awansujących

---

### Krok 4 — Warstwa narracyjna (deterministyczna)

**Dlaczego czwarty:** dopiero gdy fakty są pewne, warto je ubierać w zdania.

**Zasada — nienegocjowalna:** **żadnego LLM do generowania faktów.** Szablony
deterministyczne, wypełniane danymi z modułu `insights`. Model językowy potrafi
przekonująco zmyślić datę i to jest największe ryzyko całego produktu.

LLM można rozważyć wyłącznie do stylistycznego wygładzenia gotowego, prawdziwego zdania —
i to dopiero po zbudowaniu wszystkiego innego.

```
narrative/
├── templates_pl.py   # szablony polskie
└── renderer.py       # Fact → zdanie
```

**Kryterium ukończenia:** dla meczu reprezentacji generuje 8–10 poprawnych językowo zdań,
każde odtwarzalne z provenance.

---

### Krok 5 — Powierzchnia API

```
GET /v1/matches/{id}/preview     # pakiet faktów przedmeczowych + prognoza + uzasadnienie
GET /v1/matches/{id}/recap       # pakiet pomeczowy
GET /v1/competitions/{id}/outlook # wynik symulacji
```

Dodać: klucz API, wersjonowanie ścieżki, zapis odpowiedzi do istniejącej tabeli
`predictions`.

---

## Czego świadomie nie robimy

| Nie robimy | Dlaczego |
|---|---|
| Top 5 lig i Liga Mistrzów | Najpierw jedna ścieżka pionowa. Replikacja na kolejne rozgrywki jest tania **po** zbudowaniu maszynerii, droga przed |
| GNN | Brak ścieżki do przychodu przy jednoosobowym zespole |
| LLM do generowania faktów | Ryzyko błędu rzeczowego publikowanego pod cudzym nazwiskiem |

**Uwaga (20 sierpnia 2026):** decyzja "martj42 wystarczy, nie dodajemy kolejnych źródeł" i
"nie dostrajamy modelu, nie dodajemy cech" są **uchylone** — pionowa ścieżka (Kroki 1-5) jest
w praktyce zbudowana i zweryfikowana testami (patrz historia commitów), więc kolejny sensowny
krok to rozbudowa, nie utrzymywanie status quo. Świadomie otwieramy się na nowe źródła
statystyk (np. footballsquads.co.uk — realne, ustrukturyzowane składy turniejowe od 1950/1980
do dziś, potwierdzone as-of 2026-08-20; rsssf.org jako katalog/archiwum, nie osobne dane ponad
martj42; estadisticasdefutbol.net obecnie nieosiągalny, sprawdzone realnie przeglądarką) i na
nowe cechy modelu pod nadchodzący pełny retrening — pod warunkiem tej samej rygorystyczności co
reszta projektu: dane realne i cytowalne, nigdy zgadywane z pamięci modelu (patrz
`PROMPT_GEOCODING.md`), i żadnej zmiany w `features/builder.py` bez jawnej decyzji, tak jak przy
błędzie neutral-venue.

**Uwaga (25 sierpnia 2026):** decyzja o statystykach zespołowych (shots/corners/cards/possession)
podjęta — zostajemy przy obecnym pułapie danych. Nie kupujemy płatnego planu api-football.com pod
te statystyki, ani jako stała subskrypcja, ani jako jednorazowy zakup-ściągnij-anuluj: sprawdzone
realnie w regulaminie (api-football.com/terms, stan 2026-08-25) — dostawca nie udziela licencji na
publikację/zatrzymanie danych po zakończeniu subskrypcji, tylko na dostęp w jej trakcie, więc status
prawny danych zachowanych po anulowaniu jest niejasny, a projekt mierzy w komercyjne B2B. Retrening
idzie na tym, co realnie jest w bazie dziś plus cokolwiek doda Data Engine (footballsquads,
provenance, quality layer) — bez nowego źródła stat zespołowych. Przed samym retreningiem nadal
trzeba: (1) zaplanowane odświeżanie wyników meczów (obecnie brak automatyzacji — dane stały od
2 sierpnia, co ujawnił żywy przykład ASEAN Cup), (2) gate neutral-venue w `features/builder.py`,
(3) realny raport z `data_quality/report.py` — nikt go jeszcze nie odpalił.

**Uwaga (27 sierpnia 2026):** `home_team_name`/`away_team_name` usunięte z cech kategorycznych
modelu w `features/builder.py` — potwierdzony realny przeciek (nie hipoteza): SHAP bezpośrednio
na wytrenowanym `match_outcome` pokazał, że dla historycznie zdominowanej drużyny (Niemcy)
one-hot `away_team_name_Germany` był pojedynczą najważniejszą cechą całej predykcji, silniejszą
niż Elo/rolling, i nieczułą na `neutral_venue` (bo to surowa tożsamość, nie flaga pozycyjna jak
naprawiony wcześniej `home_is_home`). Para bardziej generyczna (Portugalia/Holandia) nie
pokazywała tego efektu w tym stopniu — potwierdza, że problem śledzi konkretną, historycznie
jednostronną drużynę, nie jest stałym artefaktem. Duży rozjazd Elo + neutralny grunt to profil
meczu grupowego mistrzostw świata — rdzeń tego, co silnik ma przewidywać, nie margines, stąd
decyzja usunięcia, nie akceptacji rezydualnego obciążenia. Ablacja (`match_outcome`, ten sam
`--n-trials 10`, ten sam podział) potwierdziła niski koszt: accuracy -0.0077, f1 -0.0062, obie
poniżej 1 punktu procentowego — usunięcie tanie, nie utrata realnego sygnału.

---

## Kolejność w jednym zdaniu

**Historia → fakty z weryfikacją → symulacje → zdania → endpoint.**
Jedna kompletna ścieżka na rozgrywkach reprezentacyjnych, gotowa przed 24 września.
Dopiero potem replikacja na piłkę klubową.

---

## Uwaga o stanie repo

Niescommitowane pozostają zmiany w `football_ai/features/player_reference.py`
i `football_ai/predictions/service.py` (spoza naszej pracy). Do przejrzenia i zamknięcia
przed rozpoczęciem Kroku 1, żeby nie mieszać ich z nowymi zmianami.
