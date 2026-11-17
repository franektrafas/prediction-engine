# Prompt do Claude Code — runda 2: wybór meczu i sekcja formy

> Poprzednia runda (`PROMPT_STREAMLIT.md`) jest zastosowana. To jest kontynuacja.
> Skopiuj wszystko poniżej linii.

---

Popraw `football_ai/streamlit_app.py`. Dwa problemy: fakty na ekranie analizy
wyglądają losowo, a wybór meczu jest chaotyczny.

Obowiązują zasady z `CLAUDE.md` i `PLAN_SILNIKA.md`. **Nie dotykaj silnika**
(`features/`, `training/`, `models/`, `predictions/service.py`). Nie zmieniaj też
logiki w `insights/` — wszystko, czego potrzebujesz, jest już policzone; problem
jest w tym, co aplikacja z tego wybiera i jak to układa.

## Problem 1 — fakty wyglądają na losowe (przyczyna, nie objaw)

Ekran analizy pokazuje wyłącznie `facts_for_match(..., top_n=10)`, czyli jedną
płaską listę posortowaną po `fact.weight` — wymieszaną z obu drużyn i ze
wszystkich 12 typów faktów naraz.

To jest źródło wrażenia losowości i ma trzy konkretne przyczyny:

1. **`weight` porównuje rzeczy nieporównywalne.** `match_count_milestone` ma na
   sztywno `weight=0.3` (patrz `insights/milestones.py`), a `h2h_never_won`
   liczy `0.1 + liczba_spotkań * 0.03`. To są dwie zupełnie różne skale
   udające jedną.
2. **Jubileusze opisują przeszły mecz, nie ten nadchodzący.**
   `match_count_milestone` odpala się tylko wtedy, gdy licznik meczów drużyny
   *przypadkiem* wypadł na okrągłej liczbie przed cutoffem, i mówi „setny mecz
   Polski był z X dnia Y". Dla zapowiedzi meczu z Bośnią to jest ciekawostka,
   nie kontekst — i dlatego czytelnik ma rację, że mu to nie pasuje.
3. **Rzeczy najbardziej meczowe przegrywają walkę o 10 miejsc.** Seria bez
   porażki i bilans bezpośredni mogą zostać wypchnięte z listy przez jubileusz
   drugiej drużyny. Co gorsza: `insights/h2h.py::h2h_matches()` zwraca pełną
   historię spotkań z kolumną `team_a_result` — i aplikacja **w ogóle jej nie
   używa**, bierze tylko wyprowadzony z niej `Fact`.

**Rozwiązanie: rozdziel dwie różne rzeczy, które dziś są jedną listą.**

**A. Kontekst strukturalny — zawsze obecny, nigdy rankingowany.** Nowa sekcja
nad prognozą, złożona z danych, które już są policzone:

- **Bilans bezpośredni** — z `h2h.h2h_matches(records_frame, home, away, as_of)`.
  Zliczone W/R/P z perspektywy gospodarza + tabela ostatnich 5 spotkań
  (data, wynik, turniej). Gdy drużyny nigdy nie grały — jedno zdanie, że to
  pierwsze spotkanie, nie pusty blok.
- **Forma: ostatnie 10 meczów** dla każdej drużyny — z
  `streaks.team_matches(records_frame, team, as_of)`, weź `.tail(10)`. Ta funkcja
  zwraca już `opponent`, `goals_for`, `goals_against`, `is_home`, `match_date`,
  więc nie licz niczego samodzielnie. Wyświetl jako rząd 10 znaczników W/R/P
  (najstarszy z lewej), a pod spodem `st.expander` z pełną tabelą: data,
  przeciwnik, wynik, dom/wyjazd.
- **Aktualne serie** — z `streaks.all_current_streaks(...)`, jako linia
  podsumowania pod znacznikami („8 meczów bez porażki"). Gdy brak serii,
  napisz wprost „brak serii", nie ukrywaj bloku.

**B. Osobna sekcja „Nietypowe fakty"** — dotychczasowa ranking­owana lista
(`facts_for_match`), ale przeniesiona **pod** prognozę i wyraźnie tak nazwana.
Wtedy jubileusz przestaje być losowy, bo jest zapowiedziany jako ciekawostka,
a nie jako główny kontekst meczu. Zachowaj provenance i wskaźnik wagi z rundy 1.
Odfiltruj z tej listy typy, które trafiły już do sekcji A
(`h2h_first_win_since`, `h2h_never_won` i typy serii), żeby to samo nie
pojawiało się dwa razy na jednym ekranie.

Nie usuwaj jubileuszy z produktu — dla dziennikarza „to będzie setny mecz" jest
publikowalne. Zmienia się tylko to, że przestają konkurować z kontekstem meczu.

## Problem 2 — wybór meczu jest chaotyczny

Dziś przed zobaczeniem czegokolwiek trzeba przejść przez pięć kontrolek jedna
pod drugą: `st.radio` (źródło) → dywizja → grupa → sezon → `st.selectbox` z
meczami sformatowanymi jako tekst → przycisk. Mecz jako linia w rozwijanej
liście to najsłabszy możliwy sposób pokazania czegoś, co ma datę, godzinę i dwie
drużyny.

**Zbuduj układ dwuekranowy (master-detail) w tym samym pliku.**

**Ekran 1 — lista meczów:**

- Dywizja i grupa jako `st.segmented_control` (nie `selectbox`) — wszystkie opcje
  widoczne naraz, wybór jednym kliknięciem.
- Sezon schowaj do `st.expander` z ustawieniami. To pole zmienia się raz na rok,
  a dziś zajmuje jedną trzecią szerokości na równi z wyborem grupy.
- Poniżej **lista meczów pogrupowana datami**. Nagłówek daty (np. „Czwartek,
  24 września"), a pod nim mecze jako osobne kontenery
  (`st.container(border=True)`), pełna szerokość, jeden pod drugim.
- Zawartość kontenera: godzina po lewej, obie drużyny pod sobą pośrodku,
  po prawej Elo obu drużyn (masz je za darmo z `_load_elo_ratings()`) i przycisk
  „Analiza". **Nie licz predykcji dla kart** — to by wymagało przeliczenia całej
  grupy przy każdym wejściu.
- Kliknięcie ustawia `st.session_state["selected_fixture"]` i woła `st.rerun()`.

**Ekran 2 — analiza:** ten sam widok co dziś, ale renderowany **zamiast** listy,
gdy `selected_fixture` jest ustawione. Na górze przycisk powrotu („← Wszystkie
mecze grupy"), który czyści klucz i wraca do listy.

**„Dowolna para"** zostaje, ale przenieś ją do `st.expander` pod listą meczów.
To jest furtka dla zaawansowanych, nie równorzędna ścieżka — dziś `st.radio` na
samej górze sugeruje, że użytkownik musi podjąć tę decyzję jako pierwszą.

## Wymagania wizualne

- Znaczniki formy: kwadrat ~22px z literą **W/R/P w środku**, nie sam kolor —
  kolor jest wzmocnieniem, nie nośnikiem informacji (daltonizm, druk, zrzut b&w).
  Litery zgodne z tabelą grupy, która już używa W/R/P.
- Kolory znaczników stonowane, w tej samej rodzinie co reszta motywu. Nie
  jaskrawa zieleń/czerwień z tabel bukmacherskich.
- Kontener meczu ma być klikalny wizualnie — obramowanie + wyraźny przycisk po
  prawej. Nie zostawiaj samego tekstu, który wygląda jak podpis.
- Kontrast tekstu min. 4.5:1, również w środku kolorowych znaczników (użyj
  ciemnego odcienia tej samej rodziny koloru, nie czerni).
- Hierarchia na ekranie analizy: nagłówek meczu → bilans i forma → prognoza →
  statystyki → nietypowe fakty → kontekst → tekst do publikacji → diagnostyka.

## Weryfikacja

1. `ruff check .` przechodzi.
2. Klik w kontener meczu przenosi na ekran analizy; przycisk powrotu wraca do
   listy z zachowanym wyborem dywizji i grupy.
3. Po wygenerowaniu analizy zmiana dowolnej kontrolki nie kasuje wyników
   (regresja z rundy 1 — sprawdź, czy nadal działa).
4. Sprawdź na trzech grupach z realnymi drużynami, nie tylko na Polsce:
   dywizja A grupa 1 (Holandia, Niemcy), dywizja A grupa 3 (Włochy, Belgia),
   dywizja B grupa 4. Interesuje mnie, czy dla każdej pary sekcja bilansu
   i formy się wypełnia.
5. Przypadek brzegowy: para, która nigdy ze sobą nie grała — sekcja bilansu
   musi pokazać sensowny komunikat, nie pustą tabelę ani wyjątek.
6. Przypadek brzegowy: drużyna z mniej niż 10 meczami w historii — rząd formy
   nie może się wysypać ani udawać 10 meczów.

## Czego nie robić

- Nie zmieniaj `insights/` — potrzebne dane już zwracają `h2h_matches()`
  i `team_matches()`.
- Nie licz predykcji dla wszystkich meczów listy naraz.
- Nie usuwaj jubileuszy ani panelu diagnostycznego.
- Nie dodawaj trzeciego poziomu zagnieżdżenia (lista → analiza → cokolwiek).
