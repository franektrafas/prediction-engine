# Prompt do Claude Code — runda 4: frekwencja i sędziowie z ostatnich meczów

> Rundy 1–3 zastosowane. Skopiuj wszystko poniżej linii.

---

Dodaj nowe źródło danych i wykorzystaj je w `football_ai/streamlit_app.py`.
Przeczytaj najpierw `CLAUDE.md` i `PLAN_SILNIKA.md`.

**Zakres — celowo wąski, nie poszerzaj go:** to nie jest kolejne źródło do
`records_frame`/`training_frame` i nie ma dotyczyć pełnej historii. Chodzi
wyłącznie o **ostatnie ~10–15 meczów każdej reprezentacji**, żeby wzbogacić
sekcję formy i mapy cieplne, które już działają. Nie dotykaj `features/`,
`training/`, `models/`, `predictions/service.py` ani logiki w `insights/`.

**Nierówne pokrycie jest z założenia akceptowalne.** Dla jednej reprezentacji
dane będą bogate, dla innej ich zabraknie w ogóle — to nie jest błąd do
naprawienia, tylko naturalna właściwość źródła. Każdy element UI zbudowany na
tych danych musi się stąd zdegradować: brakujące pole pomijasz, nie pokazujesz
placeholdera, nie blokujesz reszty karty meczu.

## Źródło: strony Wikipedii „wyniki reprezentacji"

Sprawdziłem to na żywo, nie na sucho. Wikipedia prowadzi dla większości
reprezentacji strony w formacie `„<Kraj> national football team results
(<okres>)"`, np. `Poland_national_football_team_results_(2020–present)`,
`Germany_national_football_team_results_(2020–present)`. Realna zawartość
takiej strony (sprawdzone na Polsce) to per-mecz: data, strzelcy z minutami
(w tym karne i samobóje), **frekwencja** (prawie zawsze wypełniona, łącznie
z jawnym „0" dla meczów za zamkniętymi drzwiami w czasie COVID) i **sędzia**
(wypełniony dla mniejszości meczów — z tego co widziałem, głównie przy
ważniejszych spotkaniach, nie przy sparingach — to jest właśnie ta nierówność,
o której mowa wyżej). Na dole strony jest dodatkowo **gotowa tabela
bilansów H2H** przeciwko każdemu przeciwnikowi od początku okresu strony
(P/W/D/L/GF/GA/procenty) — to nie jest główny cel tej rundy, ale potraktuj to
jako darmowy bonus do zadania w sekcji C poniżej.

**Uwaga techniczna, którą musisz sam zweryfikować:** struktura tej strony
(zbiorcza, kalendarzowa, wiele meczów na jednej stronie) różni się wizualnie
od pojedynczych bloków `{{Football box}}`, które `scrapers/sources/
uefa_nations_league.py::parse_football_box_matches()` dziś parsuje dla stron
grupowych Ligi Narodów. Nie zakładaj, że ten parser zadziała bez zmian —
sprawdź rzeczywisty wikitekst tej strony (przez ten sam mechanizm co
`fetch_wikitext()`) i albo dostosuj istniejący parser, albo napisz nowy w tym
samym stylu, jeśli szablon jest inny. Współdziel to, co da się współdzielić
(`fetch_wikitext`, `_strip_wikilinks`, `_parse_goals`), nie kopiuj na ślepo.

## A. Nowy moduł

`football_ai/scrapers/sources/team_recent_results_wikipedia.py`
(albo podobna nazwa — trzymaj się konwencji istniejących plików w tym
katalogu). Publiczne API w stylu istniejących źródeł:

- Funkcja przyjmująca nazwę drużyny (kanoniczną, przez
  `canonicalize_team_name`) i zwracająca ramkę z ostatnimi meczami:
  data, przeciwnik, frekwencja (może być `NaN`), sędzia (może być `NaN`).
- Strona nie istnieje / nie udało się jej sparsować → zwróć pusty, poprawnie
  ustrukturyzowany DataFrame, nie wyjątek — dokładnie ten sam kontrakt co
  `_load_records_frame()` w Streamlicie i `WorldFootballEloSource` przy
  błędzie sieci (`try/except` → pusty wynik + log ostrzeżenia).
- Cache: to są dane, które nie zmieniają się w trakcie sesji użytkownika —
  `@st.cache_data(ttl=...)` rzędu kilku godzin w warstwie Streamlita, żeby nie
  odpytywać Wikipedii przy każdym przeładowaniu.
- Test jednostkowy z zapisanym (fixture) fragmentem realnego wikitekstu tej
  strony — nie odpytuj żywej Wikipedii w testach.

## B. Wpięcie w `streamlit_app.py`

- Sekcja formy (ostatnie 10 meczów) na ekranie przedmeczowym: przy każdym
  meczu, dla którego mamy frekwencję i/lub sędziego, dopisz to drobnym,
  wyciszonym tekstem obok wyniku. Braku danych nie sygnalizuj wizualnie —
  po prostu nie ma tam nic dodatkowego.
- Jeśli sędzia danego, nadchodzącego meczu jest już znany (np. z terminarza
  football_api albo z samej strony Wikipedii, jeśli mecz już tam figuruje) —
  pokaż go w nagłówku analizy przedmeczowej.
- **Wątek sędziowski, do zbadania, nie do wymuszenia:** jeśli dla danej
  reprezentacji uzbiera się wystarczająco meczów z tym samym sędzią w ostatnich
  kilku latach (sprawdź realnie, ile to wychodzi — spodziewam się pojedynczych
  przypadków, nie dziesiątek), rozważ krótką notkę kontekstową typu „ten sędzia
  prowadził już N spotkań tej reprezentacji". Jeśli dane są za rzadkie, żeby to
  miało sens statystyczny (np. mniej niż 3 wspólne mecze), pomiń to całkowicie
  zamiast pokazywać notkę na podstawie jednego meczu — to nie jest fakt godny
  pokazania, tylko szum.

## C. Bonus: cross-check H2H (opcjonalnie, jeśli starczy czasu)

Tabela bilansów H2H z dołu strony Wikipedii to gotowy, niezależny od
`martj42` punkt odniesienia. Nie zastępuj nią `insights/h2h.py` — użyj jej
wyłącznie jako **testu zgodności**: dla kilku realnych par drużyn porównaj
sumę W/D/L z tej tabeli (w okresie, jaki obejmuje strona) z tym, co liczy
`h2h.h2h_matches()` na `records_frame` dla tego samego okna czasowego. Jeśli
się zgadza — nic nie zmieniaj, to tylko potwierdzenie, że dane źródłowe są
spójne. Jeśli się nie zgadza — zgłoś rozbieżność w podsumowaniu, nie
naprawiaj jej samodzielnie w tej rundzie (może to być różnica w traktowaniu
meczów towarzyskich albo strefy czasowej, wymaga decyzji, nie automatycznej
poprawki).

## Weryfikacja

1. `ruff check .` i `pytest football_ai/tests` przechodzą.
2. Uruchom dla co najmniej 3 reprezentacji: jednej z bogatym pokryciem
   (spodziewam się, że Polska taka będzie), jednej mniejszej, gdzie strona
   Wikipedii może w ogóle nie istnieć pod tą nazwą, i jednej pośredniej.
   Aplikacja ma się zachować sensownie we wszystkich trzech przypadkach, nie
   tylko w najlepszym.
3. Sprawdź mecz z realną frekwencją „0" (COVID, zamknięte drzwi) — to musi
   się wyświetlić jako 0, nie jako brak danych.
4. Potwierdź w konsoli/logu, że strona jest pobierana raz na sesję cache'a,
   nie przy każdym kliknięciu w aplikacji.

## Czego nie robić

- Nie ciągnij pełnej historii — tylko ostatnie ~10–15 meczów per drużyna.
- Nie buduj tego jako nowego wejścia do `records_frame`/treningu modelu.
- Nie pokazuj noty o sędzim opartej na jednym wspólnym meczu.
- Nie automatyzuj poprawki rozbieżności H2H z sekcji C — tylko zgłoś.
