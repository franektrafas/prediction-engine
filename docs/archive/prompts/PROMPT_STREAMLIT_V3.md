# Prompt do Claude Code — runda 3: strona główna, statystyki pomeczowe, heatmapy

> Rundy 1–2 (`PROMPT_STREAMLIT.md`, `PROMPT_STREAMLIT_V2.md`) zastosowane.
> Skopiuj wszystko poniżej linii.

---

Rozbuduj `football_ai/streamlit_app.py`. Obowiązują zasady z `CLAUDE.md`
i `PLAN_SILNIKA.md`. **Nie dotykaj silnika** (`features/`, `training/`,
`models/`, `predictions/service.py`) ani logiki w `insights/`.

Wyjątek, na który masz zgodę: możesz **dodać** nową funkcję do
`simulation/competition.py` (rozkład końcowych pozycji) — nie zmieniając
żadnej istniejącej.

## Krok 0 — zmierz pokrycie danych, zanim cokolwiek zbudujesz

Kilka zadań poniżej zależy od tego, ile realnie mamy danych. Napisz jednorazowy
skrypt (nie commituj go), który na `records_frame` i `goalscorers_frame` wypisze
procent nie-pustych wartości dla: `home_shots`, `home_shots_on_target`,
`home_corners`, `home_possession_pct`, `home_pass_pct`, `home_saves`,
`home_fouls_committed`, `home_offsides`, `home_yellow_cards` oraz `minute`
w `goalscorers`. Dodatkowo: `tournament.value_counts()` i to samo pokrycie
w rozbiciu na ostatnie 5 lat.

**Buduj tylko to, co ma realne pokrycie.** Sekcja statystyk pokazująca „brak
danych" dla ośmiu z dziesięciu pozycji jest gorsza niż jej brak. Jeżeli jakaś
statystyka ma pokrycie poniżej ~20% w ostatnich latach, nie umieszczaj jej na
stałe — schowaj do rozwijanego bloku albo pomiń. Napisz w podsumowaniu, co
zmierzyłeś i co z tego wyciąłeś.

## 1. Nagłówek i eksport

- Na samej górze, wyśrodkowana, duża nazwa: **TRAFF STATS**. Nazwa robocza —
  trzymaj ją w jednej stałej na górze pliku, żeby zmiana była jednolinijkowa.
  Pod spodem jedna cienka linia opisu, mniejszą czcionką.
- W prawym górnym rogu **„Eksportuj do PDF"** — generuje jednostronicowy dokument
  z aktualnie oglądanej analizy (nagłówek meczu, bilans, forma, prognoza,
  statystyki, nietypowe fakty z provenance). To ma być rzecz, którą dziennikarz
  wysyła dalej, więc: polskie znaki muszą działać (osadź czcionkę z polskim
  zestawem, nie licz na domyślną), stopka z datą wygenerowania i informacją,
  że liczby pochodzą z modelu, nie z zakładu.
- **Usuń sekcję „Tekst do publikacji".** Uwaga na marginesie: ona nie kosztowała
  praktycznie nic obliczeniowo (to było sklejanie stringów), więc usuwaj ją
  dlatego, że PDF jest lepszym nośnikiem tego samego — nie z powodu wydajności.

## 2. Strona główna — nadchodzące mecze

Zamiast wchodzenia od wyboru dywizji i grupy, po wejściu użytkownik ma od razu
widzieć mecze.

- Na górze **filtr rozgrywek** (radio/`st.segmented_control`) z podpisem
  „Filtruj rozgrywki". Domyślnie „Wszystkie".
- Pod spodem **najbliższe nadchodzące mecze** (ok. 10), chronologicznie,
  w kontenerach z rundy 2 (data-nagłówek, godzina, drużyny, Elo, przycisk).
- Wybór rozgrywek bez terminarza → komunikat **„Brak nadchodzących spotkań dla
  <nazwa turnieju>"**. Nie pusty ekran, nie wyjątek.
- Klik w mecz → ekran analizy przedmeczowej (już istnieje), z turniejem
  ustawionym na ten z terminarza, nie na sztywno „UEFA Nations League".

**Skąd brać terminarz — ważne.** Dziś jedynym źródłem jest
`scrapers/sources/uefa_nations_league.py` (Wikipedia, tylko Liga Narodów).
Ale `scrapers/sources/football_api.py` ma `fetch_fixtures(fixture_date,
league_id=..., next_count=...)`, a `FOOTBALL_API_KEY` jest ustawiony w `.env` —
czyli realnie da się mieć terminarz wielu rozgrywek. Zrób tak:

1. Warstwa źródeł terminarza z jednym wspólnym wyjściem (data, godzina,
   gospodarz, gość, turniej), z Ligą Narodów z Wikipedii i resztą z football_api.
2. **Agresywny cache** (`@st.cache_data(ttl=...)`, minimum kilkanaście minut) —
   to jest płatne API i strona główna nie może go odpytywać przy każdym
   przeładowaniu skryptu.
3. Brak klucza albo błąd sieci → degraduj do samej Ligi Narodów i pokaż o tym
   dyskretną informację. Nigdy nie wywalaj aplikacji.

## 3. Ekran przedmeczowy — zmiany

- **Usuń „Przewidywany wynik"** (dokładny score). Zamiast niego **mapa cieplna
  prawdopodobieństwa wyników** — siatka 0–5 goli gospodarza × 0–5 goli gościa,
  każda komórka to prawdopodobieństwo tego dokładnego rezultatu, liczone
  z rozkładu Poissona na przewidywanych golach (dokładnie tak, jak robi to
  `simulation/monte_carlo.py`). Podświetl najbardziej prawdopodobną komórkę
  i wypisz pod spodem 3 najbardziej prawdopodobne wyniki z procentami.
  To rozwiązuje ten sam problem uczciwiej: jeden „przewidywany wynik 2:1"
  sugeruje pewność, której model nie ma; siatka pokazuje realny rozrzut.
- **Przycisk „Zobacz perspektywy grupy"** — przenosi do sekcji symulacji
  z ustawioną dywizją i grupą tego meczu. Jeśli mecz nie należy do żadnej grupy
  (towarzyski), nie pokazuj przycisku zamiast pokazywać nieaktywny.

## 4. Ekran pomeczowy — przebuduj

- **Domyślnie lista ostatnio rozegranych meczów** (ok. 10, chronologicznie od
  najnowszego), tak samo w kontenerach jak strona główna, ale z wynikiem
  zamiast godziny.
- **Dwa niezależne filtry**, oba opcjonalne:
  - rozgrywki (jak na stronie głównej),
  - **drużyna** (`st.selectbox` z wyszukiwaniem).

  Zachowanie: żaden filtr → ostatnie mecze ze wszystkich rozgrywek; sam turniej →
  ostatnie mecze tego turnieju; sama drużyna → jej ostatnie 10 meczów wszędzie;
  oba → przecięcie. To zwykły filtr na `records_frame`, bez nowych źródeł danych.
- **Pokaż realne statystyki meczowe, jeśli są.** `records_frame` zawiera
  `home_/away_`: `shots`, `shots_on_target`, `corners`, `yellow_cards`,
  `red_cards`, `possession_pct`, `fouls_committed`, `offsides`, `total_passes`,
  `pass_pct`, `saves`. Wyświetl jako pary poziomych pasków gospodarz–gość
  (ten sam komponent, którego używasz na ekranie przedmeczowym). Statystyki
  bez danych dla tego meczu **pomiń całkowicie**, nie renderuj pustych wierszy.
  Gdy mecz nie ma żadnych statystyk — jedno zdanie wyjaśnienia (dane statystyczne
  są dostępne głównie dla nowszych meczów), nie pusta sekcja.
- **Strzelcy** — `goalscorers_frame` ma `scorer`, `minute`, `penalty`,
  `own_goal`. Wypisz gole chronologicznie z minutą i oznaczeniem karnego
  i samobójczego. To jest tanie i bardzo naturalne dla relacji pomeczowej.

## 5. Perspektywy grupy — wstęp i mapa pozycji

- **Krótki wstęp na górze sekcji**, 2–3 zdania, bez żargonu. Ma wyjaśnić, co się
  tu dzieje i skąd biorą się liczby. „Liczba symulacji Monte Carlo" nic nie mówi
  komuś spoza branży — napisz to po ludzku, np. że rozgrywamy pozostałe mecze
  grupy dziesiątki tysięcy razy, losując wyniki zgodnie z prognozą modelu,
  i liczymy, jak często każda drużyna kończy na miejscu premiowanym awansem.
  Suwak liczby symulacji zostaw, ale przenieś go do ustawień w `st.expander`.
- **Zachowaj widoczną informację o uproszczeniu tiebreaku** (brak head-to-head),
  którą dodała runda 1.
- **Dodaj mapę cieplną rozkładu miejsc**: drużyny w wierszach, miejsca 1–4
  w kolumnach, komórka = jak często drużyna kończy na tym miejscu. To znacznie
  bogatsze niż jeden procent awansu — pokazuje różnicę między „drugie miejsce
  prawie pewne" a „albo pierwsze, albo czwarte".
  Dodaj w tym celu **nową** funkcję w `simulation/competition.py` (np.
  `simulate_final_positions`) zbudowaną z istniejących `run_simulations`
  i `compute_standings` — nie modyfikuj `simulate_group_qualification`.
  Dopisz do niej test w `football_ai/tests/unit/test_simulation_competition.py`
  sprawdzający, że każdy wiersz sumuje się do 1.0.

## 6. Trzecia mapa cieplna — profil minutowy goli

Na ekranie przedmeczowym, w sekcji formy: dla obu drużyn rozkład goli
**strzelonych i straconych** w przedziałach 15-minutowych (1–15, 16–30, ...,
76–90+), z `goalscorers_frame.minute`, na ostatnich ~20 meczach każdej drużyny.

To jest wdzięczne dziennikarsko („Bośnia traci połowę goli po 70. minucie")
i całkowicie weryfikowalne. **Zbuduj to tylko, jeśli pomiar z Kroku 0 pokaże
sensowne pokrycie kolumny `minute`** — jeśli nie, pomiń i napisz o tym.

## 7. Estetyka

- Mapy cieplne: **jednokolorowa skala** (jasny → ciemny w rodzinie koloru
  akcentu), nie tęcza. Wartości wypisane w komórkach jako tekst — kolor jest
  wzmocnieniem, nie jedynym nośnikiem. Tekst w ciemnych komórkach musi
  pozostać czytelny (przełącz na jasny powyżej progu wypełnienia).
- Siatka wyników: osie podpisane nazwami drużyn, nie „home"/„away".
- Nagłówek TRAFF STATS: duży, wyśrodkowany, ale bez cienia i gradientu —
  spójny ze stonowanym motywem z `.streamlit/config.toml`.
- Filtry na stronie głównej i pomeczowej mają wyglądać identycznie — to ta sama
  czynność w dwóch miejscach.
- Kontrast min. 4.5:1 wszędzie, łącznie z komórkami map cieplnych.

## 8. Weryfikacja

1. `ruff check .` przechodzi; `pytest football_ai/tests` przechodzi.
2. Strona główna działa **bez** `FOOTBALL_API_KEY` (wyłącz go tymczasowo) —
   degraduje do Ligi Narodów, nie wywala się.
3. Wybór rozgrywek bez terminarza pokazuje komunikat z nazwą turnieju.
4. Ekran pomeczowy: sprawdź mecz z pełnymi statystykami i mecz sprzed 1950 roku
   (żadnych statystyk) — oba muszą wyglądać sensownie.
5. Filtr drużyny bez filtra rozgrywek zwraca ostatnie 10 meczów tej drużyny.
6. Siatka wyników: prawdopodobieństwa sumują się do ~1.0 (z uwzględnieniem
   ogona powyżej 5 goli — napisz w podpisie, ile masy jest poza siatką).
7. PDF: wygeneruj dla meczu z polskimi nazwami (np. Bośnia i Hercegowina) i
   sprawdź, czy „ś", „ż", „ó" renderują się poprawnie, a nie jako puste kwadraty.
8. Mapa pozycji: każdy wiersz sumuje się do 100%.

## 9. Czego nie robić

- Nie odpytuj football_api przy każdym przeładowaniu skryptu.
- Nie licz predykcji dla wszystkich meczów na liście.
- Nie renderuj pustych wierszy statystyk „dla kompletności".
- Nie zmieniaj `simulate_group_qualification` ani niczego w `insights/`.
- Nie dodawaj tęczowych skal kolorów.
