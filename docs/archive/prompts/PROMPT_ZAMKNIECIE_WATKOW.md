# Prompt do Claude Code — domknięcie otwartych wątków (silnik, squady, travel fatigue, retrening)

> To jest ta faza, w której masz pełną swobodę zmian w silniku —
> `features/builder.py`, `training/`, `footballsquads.py`, `streamlit_app.py`.
> Wcześniejsze prompty (np. `PROMPT_HOME_ADVANTAGE_CHECK.md`) kazały nie
> ruszać kodu bez osobnej rozmowy — to było celowe na etapie diagnostyki,
> teraz to nieaktualne. Nie pytaj o zgodę przed każdą zmianą, rób. Jedyne
> rzeczy, które zostają nienegocjowalne, są wypisane na końcu w sekcji
> "Twarde zasady, które zostają" — to nie są bramki ostrożności tego
> promptu, tylko strukturalne reguły całego projektu z `CLAUDE.md`.
>
> To jest wielowątkowe zadanie z realną kolejnością zależności — trzymaj
> się numeracji, nie rób punktu 5 (retrening) przed 1-4. W środku
> zweryfikowany (nie zgadywany) aktualny stan kodu na 25 sierpnia 2026,
> więc nie musisz tego ponownie odkrywać, ale nie traktuj tego jako
> zamrożonej prawdy — jeśli coś się nie zgadza z tym co widzisz, wierz
> kodowi, nie temu dokumentowi. Skopiuj wszystko poniżej linii.

---

## Kontekst

Kilka wątków wisi otwartych jednocześnie: gate neutral-venue, Club Cohesion
Index w danych o składach, Travel Fatigue Index w Streamlicie, realny raport
jakości danych, i sam retrening. Zamiast domykać je pojedynczo i osobno
retrenować po każdym, robimy to w jednym przejściu — bo i tak wchodzimy w
fazę retreningu, więc taniej jest zebrać wszystkie zmiany silnika i
cech razem, jeden retrening na końcu, nie kilka.

Osobno: decyzja o statystykach zespołowych (shots/corners/cards/possession)
już zapadła i jest zamknięta — `PLAN_SILNIKA.md`, notatka z 25 sierpnia
2026: zostajemy przy obecnym pułapie danych, bez płatnego api-football.com.
Nie wracaj do tego tematu w ramach tego zadania.

Osobno: zaplanowane odświeżanie danych meczowych ma już własny, oddzielny
prompt — `PROMPT_DATA_REFRESH.md`. Jeśli jeszcze nie wykonany, wykonaj go
**przed** krokiem 5 (retrening) poniżej, żeby trenować na aktualnych
danych, nie na tych sprzed trzech tygodni. Nie duplikuj tamtej pracy tutaj.

---

## 1. Gate neutral-venue w `features/builder.py`

**Zweryfikowany stan:** `neutral_venue` istnieje jako surowa kolumna
(`BASE_MATCH_COLUMNS`, linia 26) i przechodzi przez cały pipeline, ale
nic go nie czyta. `_add_rolling_features()` (linie 209-235) dzieli
wyłącznie po `is_home` — statycznej etykiecie z arkusza meczowego:

```python
venue_shifted = engineered.groupby(["team_name", "is_home"])[metric].shift(1)
rolling_venue = venue_shifted.groupby(
    [engineered["team_name"], engineered["is_home"]]
).rolling(window=window, min_periods=1)
```

Efekt: mecz na neutralnym gruncie (MŚ, finały Euro, mecze LN na neutralnym
terenie) wciąż liczy jednej drużynie "formę u siebie", a drugiej "formę na
wyjeździe", dokładnie jakby to był prawdziwy mecz domowy — mimo że żadna
z nich nie gra u siebie.

**Zadanie:** wprowadź "away-split" dla meczów neutralnych — obie drużyny na
neutralnym gruncie mają zasilać wyłącznie rolling/away bucket, nie
home bucket. Praktycznie: pochodna flaga typu
`effective_is_home = is_home & (~neutral_venue)` użyta w grupowaniu zamiast
gołego `is_home`, tak żeby na `neutral_venue=True` obie strony trafiały pod
`False`. Zastosuj konsekwentnie wszędzie, gdzie `is_home` steruje
podziałem home/away w rolling/recency/h2h (nie tylko w jednej funkcji —
sprawdź `_add_recency_features` i `_add_h2h_features` też, jeśli mają
analogiczny podział).

Task #8 w backlogu ("Empirycznie sprawdzić swap neutral→away-split bez
retreningu") już to zwalidował empirycznie bez zmiany kodu — jeśli
znajdziesz w historii commitów albo w plikach roboczych ten skrypt
diagnostyczny, zweryfikuj się względem niego. Jeśli nie, kieruj się
kierunkiem opisanym wyżej i zweryfikuj przez ponowne odpalenie
`PROMPT_HOME_ADVANTAGE_CHECK.md` (krok 6 poniżej) po retreningu.

---

## 2. Club Cohesion Index w danych o składach

**Zweryfikowany stan:** plik to `football_ai/scrapers/sources/footballsquads.py`
(nie `features/footballsquads.py`). `_squad_summary()` (linie 260-301)
liczy dziś: średni wiek, średni wzrost, liczebność pozycji, i
`squad_continuity_ratio` — ale **nic związanego z klubami**. Kolumna
`Current Club` istnieje w surowych danych i przechodzi nietknięta przez
`_parse_squad_text()` (linie 135-203) aż do `players_frame`, ale nigdy nie
jest normalizowana ani agregowana. Specyfikacja indeksu jest opisana
wyłącznie w `PROMPT_FOOTBALLSQUADS.md` (linie 87, 90-114) — nigdy nie
zaimplementowana w kodzie.

**Zadanie:**
1. Znormalizuj kolumnę `Current Club` (spójne nazwy klubów — sprawdź czy
   jest już gdzieś w projekcie tabela aliasów klubów analogiczna do
   `config/team_aliases.py` dla reprezentacji; jeśli nie ma, zrób
   minimalną normalizację, nie buduj osobnego wielkiego modułu aliasów pod
   to jedno zadanie).
2. Policz indeks zgodnie ze specyfikacją z `PROMPT_FOOTBALLSQUADS.md`:
   grupowanie graczy po klubie, waga per klaster = C(n,2) par graczy w tym
   samym klubie (nie płaska liczba par), opcjonalnie osobno per grupa
   pozycji. Dodaj wynik do `_squad_summary()` obok istniejących pól.
3. **Sprawdź i domknij, czy istniejące już indeksy (wiek/wzrost/continuity)
   w ogóle trafiają do ramki treningowej.** Z tego co widać w kodzie,
   `footballsquads.py` liczy je, ale nie jest jasne, czy są gdziekolwiek
   merge'owane do `matches`/`build_features()` — jeśli nie są, to
   podpięcie tego merge'a (po drużynie + turnieju/dacie) jest częścią tego
   zadania, nie osobnym wątkiem na później. Bez tego Club Cohesion Index
   policzy się poprawnie, ale model go nigdy nie zobaczy.

---

## 3. Travel Fatigue Index w Streamlicie

**Zweryfikowany stan:** `_render_travel_fatigue` nie istnieje nigdzie w
kodzie — jedyne wystąpienie tej nazwy to komentarz w docstringu
`_haversine_km()` (`streamlit_app.py:970-989`), która sama jest gotowa i
poprawna (self-test w docstringu), ale nigdzie nie wywoływana. Moduł
geokodowania, `football_ai/data/geocoding.py`, jest **kompletny i gotowy
do użycia** — `coordinates_for_city_country(city, country)` i
`coordinates_for_stadium(stadium_name)`, oparte na `geonamescache` plus
tabela 67 realnych stadionów Ligi Narodów (zweryfikowana 2026-08-19). Nie
był dotąd importowany nigdzie poza własnym testem jednostkowym.

**Zadanie:** zbuduj `_render_travel_fatigue` w `streamlit_app.py` — dla
danego meczu policz dystans (via `_haversine_km` + `geocoding.py`) między
poprzednim miejscem gry każdej drużyny a miejscem nadchodzącego meczu,
pokaż to jako czytelny kontekst przedmeczowy (km od ostatniego meczu,
liczba dni odpoczynku jeśli już liczone gdzie indziej, ewentualnie zmiana
strefy czasowej).

**To zostaje kontekstem, nie cechą modelu — to nie jest nowa bramka tego
promptu, tylko już wcześniej ustalony zakres (task #7).** Nie merge'uj
tego do `features/builder.py` ani do ramki treningowej. Powód wciąż ten
sam: to log presentation/insight layer, nie zweryfikowany feature
inżynieryjny — inaczej niż Club Cohesion Index z punktu 2, które jest
świadomie projektowane jako cecha modelu.

---

## 4. Realny raport `data_quality/report.py`

**Zweryfikowany stan:** moduł jest w pełni zaimplementowany, ma CLI
(`argparse`, linie 183-212), liczy pokrycie shots/shots_on_target/corners/
cards/possession/xg per dekada i per turniej. Nikt go jeszcze nie odpalił
naprawdę.

**Zadanie:** odpal

```bash
python -m football_ai.data_quality.report \
  --records-frame football_ai/data/processed/international_results_full_history.parquet \
  --include-goalscorers
```

i zgłoś realne liczby — nie tylko "zadziałało". Konkretnie: pokrycie per
stat per dekada (czy praca Data Engine z ostatnich dni faktycznie coś
poprawiła dla shots/corners/cards, czy tylko zbudowała infrastrukturę bez
nowych danych), i wszystkie zgłoszone `QualityIssue` z `validators.py` i
`provenance_checks.py`. To ma wpływ na to, co realistycznie można obiecywać
w warstwie faktów/narracji (Krok 2/4 z `PLAN_SILNIKA.md`) — jeśli pokrycie
jest nadal niskie dla starszych dekad, to trzeba to świadomie uwzględnić
tam, nie milczeć.

---

## 5. Połączony retrening

**Dopiero po 1, 2, 4 wyżej, i po `PROMPT_DATA_REFRESH.md` jeśli jeszcze nie
wykonany.**

Komenda (zweryfikowana względem aktualnego `argparse` w
`training/orchestrator.py` — flaga to `--targets`, nie `--target-names`
jak błędnie w starszym `PROMPT_TRENING_WZNOWIENIE.md`):

```bash
python -m football_ai.training.orchestrator \
  --matches-path <ścieżka do aktualnej ramki treningowej> \
  --version <nowy tag wersji> \
  --targets <lista targetów po przecinku> \
  --n-trials <int> \
  --optuna-storage <ścieżka lub "">
```

Domyślnie trening czyta `international_results_modern_with_elo.parquet`
(via `settings.inference_data_path`) jeśli `--matches-path` pominięte —
**zdecyduj świadomie**, czy to dalej właściwy plik po zmianach z punktów
1-2, czy potrzebny jest nowy zmergowany plik (np. z Club Cohesion Index
domerge'owanym) i podaj go explicite przez `--matches-path`. Nie polegaj
domyślnie na pliku sprzed tych zmian.

Pamiętaj o twardej zasadzie z `CLAUDE.md`: pre-~1990 mecze tylko do
`records_frame`/rekordów, nigdy do treningu (`international_results_full_history.parquet`
≠ ramka treningowa).

---

## 6. Ponowna diagnostyka home-advantage po retreningu

Odpal ponownie skrypt z `PROMPT_HOME_ADVANTAGE_CHECK.md` (ten sam kod, na
nowo wytrenowanym modelu) i porównaj `swing_neutral` / `swing_real_home`
sprzed i po. Zgłoś czy fix z punktu 1 faktycznie zmienił zachowanie modelu,
nie tylko dane wejściowe.

---

## Co zgłosić z powrotem

Dla każdego z punktów 1-6 osobno:
1. **Neutral-venue:** dokładnie co zmieniono i gdzie (pliki/funkcje), i
   wynik ponownej diagnostyki z punktu 6.
2. **Club Cohesion Index:** czy istniejące indeksy squadów w ogóle trafiały
   do treningu przed tym zadaniem (zaskakujące, jeśli nie), i czy Cohesion
   Index po dodaniu realnie dociera do `build_features()`.
3. **Travel Fatigue:** zrzut ekranu/opis wyglądu w Streamlicie dla
   przykładowego meczu, potwierdzenie że nie jest w `features/builder.py`.
4. **Raport jakości:** pełne liczby pokrycia per dekada per stat, lista
   `QualityIssue`.
5. **Retrening:** który plik faktycznie użyty jako `--matches-path` i
   dlaczego, metryki per target (F1/RMSE na train/val/test tak jak
   standardowo loguje orchestrator).
6. **Home-advantage:** liczby przed/po.

## Twarde zasady, które zostają (strukturalne, nie ostrożnościowe)

- Zero losowych splitów. Wszystko przez `rolling_positional_splits`/
  `chronological_train_validation_test_split` z `training/splitters.py`.
  Jeśli dotykasz `tuning.py` albo `stacking_ensemble.py`, `cv=` musi dalej
  budować się stamtąd, nie z domyślnego (Stratified)KFold sklearn.
- `records_frame` (pełna historia od 1872) nigdy nie wchodzi bezpośrednio
  do treningu — tylko do rekordów/faktów.
- Travel Fatigue Index zostaje kontekstem, nie cechą modelu (patrz punkt
  3) — to jedyne miejsce w tym zadaniu, gdzie świadomie NIE łączymy nowej
  pracy z silnikiem.
- `_wide_merge`'a logika chroniąca przed kolizją nazw kolumn
  (`_drop_columns_shadowing_source`) musi zostać zachowana przy każdym
  nowym merge'u (np. Club Cohesion Index) — nie wprowadzaj nowej kolumny o
  nazwie, która już istnieje w `matches`, bez tej samej ochrony.
