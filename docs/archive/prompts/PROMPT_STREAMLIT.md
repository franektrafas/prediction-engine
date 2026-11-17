# Prompt do Claude Code — przebudowa `football_ai/streamlit_app.py`

> Skopiuj wszystko poniżej linii i wklej do Claude Code w katalogu repo.

---

Przebuduj `football_ai/streamlit_app.py`. Aplikacja działa, ale nie pokazuje
większości tego, co silnik już potrafi, ma jeden poważny błąd użytkowy i wygląda
jak domyślny Streamlit.

Przeczytaj najpierw `CLAUDE.md` i `PLAN_SILNIKA.md`. Obowiązują wszystkie zasady
projektu — w szczególności: **żadnego LLM do generowania faktów**, każdy fakt musi
nieść `provenance`, i **nie dotykaj silnika** (`features/`, `training/`, `models/`,
`predictions/service.py`). To jest praca wyłącznie w warstwie prezentacji.

Odbiorca: **dziennikarz sportowy**, nie analityk i nie deweloper. Ekran ma
odpowiadać na jedno pytanie — „czy mogę to opublikować pod swoim nazwiskiem".
Wszystko, co nie służy temu pytaniu, jest drugorzędne.

## A. Błąd użytkowy — napraw najpierw

Obie zakładki używają wzorca `if not st.button(...): return`. W Streamlicie każda
interakcja z dowolnym widżetem przeładowuje skrypt, więc **wyniki znikają**, gdy
tylko użytkownik zmieni cokolwiek po ich wygenerowaniu (np. przestawi liczbę
symulacji albo zmieni datę). Przy demo na żywo to wygląda na zepsutą aplikację.

Przenieś wyniki do `st.session_state` i renderuj je niezależnie od tego, czy
przycisk został właśnie kliknięty. Przy okazji dodaj `@st.cache_data` na wyniku
predykcji kluczowanym `(home, away, match_date, tournament)` — dziś każde
kliknięcie liczy od nowa.

## B. Czego brakuje, a silnik to ma

To jest sedno zadania. Każdy punkt to istniejąca, przetestowana funkcjonalność,
której aplikacja po prostu nie pokazuje.

1. **`prediction.model_versions`** (dict per target) — aplikacja pokazuje tylko
   zbiorcze `model_version`. Bez tego nie odróżnisz „model to policzył" od
   „`_heuristic_prediction` zgadło z Elo, bo dla tego targetu nie ma wytrenowanego
   modelu". Pokaż per target, w zwijanym panelu na dole (`st.expander`), i wyraźnie
   oznacz targety schodzące na heurystykę. To jest element diagnostyczny — musi
   być zawsze dostępny, ale nie może dominować ekranu.

2. **`prediction.match_context`** — kontuzje, wartość rynkowa składu, forma
   bramkarza/napastnika. Dziś w ogóle nieużywane. Przeczytaj docstring klasy
   `MatchContext` w `predictions/schemas.py` i uszanuj go: te pola mają **zerową
   ważność w modelu** (potwierdzone ablacją) i muszą być podpisane wprost jako
   kontekst informacyjny, nie jako coś, co model uwzględnił. Osobna sekcja,
   wizualnie odseparowana od liczb modelu.

3. **`fact.provenance`** — lista `match_id`, z których powstał fakt. To jest
   *najważniejszy* element całego produktu i dziś nie ma go w UI wcale. Każde
   zdanie musi dać się rozwinąć do listy meczów źródłowych (data, drużyny, wynik —
   dociągnij je z `records_frame` po `match_id`). Bez tego to jest czarna skrzynka,
   a cała przewaga nad kursami bukmacherskimi znika.

4. **`fact.fact_type` i `fact.weight`** — dziś fakty lecą płaską listą punktów.
   Jest 12 typów (`win_streak`, `h2h_never_won`, `biggest_win`,
   `player_goal_milestone`, ...). Pogrupuj po typie albo posortuj po wadze
   z czytelnym wskaźnikiem istotności — dziennikarz musi w 3 sekundy zobaczyć,
   który fakt jest najmocniejszy.

5. **Tryb „po meczu" (recap)** — logika jest w `api/routes/matches.py::match_recap`,
   w Streamlicie jej nie ma. Dodaj trzecią zakładkę: realny wynik + fakty
   z cutoffem `as_of = data + 1s` (przeczytaj komentarz w tamtej funkcji, żeby nie
   zgubić faktów, które sam ten mecz stworzył).

6. **Scenariusze „co jeśli"** — `simulation/scenarios.py::what_if_fixed_result`
   jest zbudowane i nieodsłonięte. W zakładce symulacji: wybierz jeden
   nierozegrany mecz, ustaw wynik, pokaż szanse awansu przed i po. To jest
   treść niedostępna w żadnym darmowym źródle — dokładnie to, co ma sprzedawać
   ten produkt.

7. **Eksport tekstu** — produktem są *zdania*, a nie da się ich dziś skopiować
   inaczej niż zaznaczaniem myszą. Dodaj blok z gotowym tekstem (`st.code`, ma
   wbudowany przycisk kopiowania) i `st.download_button` do `.md`.

8. **Polskie nazwy drużyn** — interfejs jest po polsku, a drużyny wyświetlają się
   po angielsku („Poland", „Sweden"). `narrative/team_names_pl.py::nominative_pl()`
   już to rozwiązuje. Wyświetlaj po polsku, trzymając angielską nazwę kanoniczną
   jako klucz wewnętrzny (nie zmieniaj tego, co idzie do silnika).

9. **Zastrzeżenie o tiebreaku** — `simulation/competition.py` ma udokumentowane
   uproszczenie: brak head-to-head w rozstrzyganiu remisów punktowych. Przy tabeli
   grupy musi być widoczny przypis. Produkt sprzedający weryfikowalność nie może
   ukrywać znanych ograniczeń.

## C. Estetyka

Dziś: brak `.streamlit/config.toml` (czyli domyślny motyw), emoji w tytule,
wszystkie nagłówki jako `st.markdown("**pogrubienie**")`, wszystkie wykresy
w jednym płaskim niebieskim, statystyki jako surowa tabela.

- Załóż `.streamlit/config.toml` z własnym motywem. Paleta stonowana,
  redakcyjna — ciemny granat/grafit + jeden kolor akcentu. Nie „dashboard
  bukmacherski".
- **1X2 to trzy różne kategorie semantyczne** i nie mogą mieć jednego koloru.
  Wygrana gospodarza / remis / wygrana gościa — trzy odrębne, ale nie krzykliwe.
  Nie polegaj wyłącznie na kolorze: procent zawsze widoczny jako tekst.
- Prawdziwa hierarchia zamiast pogrubień: `st.header`/`st.subheader`,
  `st.container(border=True)` do oddzielania sekcji. Kolejność czytania musi być
  jednoznaczna: mecz → prognoza → fakty (z rozwijanym provenance) → kontekst →
  panel diagnostyczny.
- Rząd metryk miesza dziś typy (wynik, xG, %, string z wersją modelu). Rozdziel:
  liczby meczowe osobno, wersja modelu należy do panelu diagnostycznego.
- Statystyki meczowe (rożne/kartki/strzały/posiadanie) czytelniej jako pary
  poziomych pasków gospodarz-vs-gość niż jako `st.dataframe`.
- Kontrast tekstu min. WCAG AA (4.5:1); sprawdź też opisy na wykresach.
- Wykresy Plotly: usuń domyślną ramkę i tło, zsynchronizuj czcionkę z motywem.

## D. Weryfikacja przed zakończeniem

1. `ruff check .` przechodzi.
2. Uruchom `streamlit run football_ai/streamlit_app.py` i przeklikaj **wszystkie
   trzy** zakładki. Po wygenerowaniu wyników zmień dowolny widżet i potwierdź, że
   wyniki **nie znikają** (to jest test na punkt A).
3. Przetestuj na trzech różnych meczach, nie tylko Polska–Szwecja — użyj realnych
   par z Ligi Narodów 2026/27: **Holandia–Niemcy**, **Włochy–Belgia**,
   **Anglia–Hiszpania** (24–26 września). Chodzi o wyłapanie problemów
   z kanonicalizacją nazw i o sprawdzenie, czy dla każdej pary w ogóle znajdują
   się fakty.
4. Sprawdź przypadek brzegowy: para drużyn, które nigdy ze sobą nie grały —
   pusta lista faktów musi dać sensowny komunikat, nie pusty ekran ani wyjątek.
5. Potwierdź, że dla co najmniej jednego faktu rozwinięcie provenance pokazuje
   realne mecze zgodne z treścią zdania.

## E. Czego świadomie nie robić

- Nie dotykaj `features/`, `training/`, `models/`, `predictions/service.py`.
- Nie dodawaj LLM-a do generowania ani „wygładzania" faktów.
- Nie buduj dashboardu z kilkunastoma wykresami. Jeden mecz, jeden ekran,
  jasna hierarchia.
- Nie ukrywaj faktu, że część targetów schodzi na heurystykę — panel
  diagnostyczny ma to pokazywać, nie maskować.
