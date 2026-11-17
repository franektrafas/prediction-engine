# Prompt do Claude Code — runda 5: mapa cieplna formy zamiast tekstu

> Rundy 1–4 zastosowane. Skopiuj wszystko poniżej linii.

---

Popraw prezentację danych z rundy 4 (frekwencja, sędzia) i dodaj właściwą
mapę cieplną formy w `football_ai/streamlit_app.py`. Przeczytaj najpierw
`CLAUDE.md` i `PLAN_SILNIKA.md`. Nie dotykaj `features/`, `training/`,
`models/`, `predictions/service.py` ani `insights/`.

## Diagnoza — dlaczego frekwencja/sędzia nie nadają się na heatmapę

Heatmapa potrzebuje siatki dwuwymiarowej z sensowną skalą koloru w każdej
komórce. Frekwencja to jedna liczba na mecz w czasie — to jest wykres
słupkowy, nie mapa cieplna, i tak to zostaw (patrz sekcja B). Sędzia to
tekst, nie liczba, i przy obecnym zakresie (ostatnie ~10–15 meczów jednej
drużyny) ten sam sędzia praktycznie nigdy się nie powtórzy na tyle razy, żeby
zbudować z tego cokolwiek statystycznie sensownego — zostaw to jako tekst
przy meczu, tak jak jest, i nie próbuj z tego robić wykresu w tej rundzie.

**Właściwym kandydatem na heatmapę są realne statystyki meczowe, które już
macie w `records_frame`** (`home_/away_shots`, `shots_on_target`, `corners`,
`yellow_cards`, `red_cards`, `possession_pct`, `fouls_committed`, `offsides`,
`pass_pct`, `saves` — potwierdzone w pliku `international_results_full_
history.parquet`). Runda 3 kazała zmierzyć pokrycie tych kolumn (Krok 0) —
jeśli ten pomiar nie został jeszcze zrobiony albo wyniku nie ma pod ręką,
zrób go teraz jako pierwszy krok tej rundy, bo od niego zależy, które wiersze
poniższej mapy w ogóle da się zbudować.

## A. Mapa cieplna formy — wiersze metryk × ostatnie 10 meczów

Na ekranie przedmeczowym, w sekcji formy każdej drużyny, dodaj siatkę:
kolumny to ostatnie 10 meczów (najstarszy z lewej, najnowszy z prawej — ta
sama kolejność co istniejący rząd znaczników W/R/P), wiersze to metryki.
Buduj **tylko te wiersze, które mają realne pokrycie w Kroku 0** — jeśli
`possession_pct` jest puste dla większości meczów danej drużyny, ten wiersz
dla niej pomiń całkowicie, nie pokazuj pustego rzędu.

Proponowane wiersze, w kolejności ważności:
1. Wynik (W/R/P) — to już macie, tylko przenieś kolorystykę do wspólnej skali.
2. Gole strzelone.
3. Gole stracone.
4. Posiadanie piłki (%) — jeśli pokrycie wystarczające.
5. Strzały celne — jeśli pokrycie wystarczające.
6. Kartki (żółte + czerwone łącznie) — jeśli pokrycie wystarczające.

**Brakująca wartość dla pojedynczego meczu w wierszu, który generalnie ma
dobre pokrycie:** ta jedna komórka zostaje pusta/wyszarzona z podpisem
„brak", wiersz się nie chowa. To inny przypadek niż cały wiersz bez pokrycia
dla drużyny — rozróżnij te dwa scenariusze.

Skalowanie koloru: **per wiersz, nie globalnie** — najciemniejszy/najjaśniejszy
odcień w wierszu „gole strzelone" ma odzwierciedlać rozrzut w tych 10 meczach
tej drużyny, nie być porównywalny między drużynami czy między metrykami.
Jedna rodzina kolorów (paleta z `.streamlit/config.toml`), rosnąca
intensywność = wyższa wartość, z wyjątkiem wiersza „wynik", który zostaje
trójkolorowy (W/R/P) jak dziś, i wiersza „gole stracone"/„kartki", gdzie
warto rozważyć odwróconą intensywność (mniej = lepiej) — podpisz to wprost
w micro-caption pod siatką, żeby nie było niejednoznaczne.

## B. Frekwencja jako wykres słupkowy

Zamiast dopisku tekstowego przy każdym meczu: mały wykres słupkowy pod
sekcją formy, jeden słupek na mecz z ostatnich ~10, wysokość = frekwencja.
Mecz bez danych o frekwencji → brak słupka w tym miejscu (przerwa), nie
słupek zerowy — zerowa frekwencja (prawdziwe „mecz za zamkniętymi drzwiami")
i brak danych to dwie różne rzeczy i muszą się różnić wizualnie.

Sędzia zostaje tekstem przy nagłówku meczu, tak jak wpięte w rundzie 4 —
nic tu nie zmieniaj.

## C. Wątek sędziowski — zakres na później, nie teraz

Notatka do backlogu, nie zadanie na tę rundę: sensowna wizualizacja
„tendencji sędziowskich" (np. średnia liczba kartek na mecz danego sędziego)
wymagałaby agregacji sędziego **przez całą bazę meczów, nie przez ostatnie
10–15 jednej drużyny** — to zmienia zakres z „wzbogacenie formy jednej
reprezentacji" na „nowy wymiar analityczny przez cały zbiór danych". Zostaw
to jako osobną decyzję na przyszłość, nie wciągaj automatycznie w tę rundę.

## Weryfikacja

1. `ruff check .` i `pytest football_ai/tests` przechodzą.
2. Sprawdź na drużynie z dobrym pokryciem statystyk i na takiej, gdzie
   Krok 0 pokazał słabe pokrycie — druga powinna pokazać mniej wierszy
   mapy, nie pustą albo zepsutą siatkę.
3. Wykres frekwencji: sprawdź mecz z prawdziwym zerem (COVID) i mecz bez
   żadnych danych o frekwencji — muszą wyglądać inaczej.
4. Kontrast tekstu w komórkach map cieplnych min. 4.5:1, także w ciemnych
   komórkach (jaśniejszy tekst powyżej progu wypełnienia).

## Czego nie robić

- Nie buduj heatmapy z frekwencji ani z sędziego.
- Nie skaluj kolorów globalnie między drużynami/metrykami — tylko per wiersz.
- Nie zaczynaj wątku sędziowskiego z sekcji C bez wyraźnej decyzji, że
  poszerzamy zakres poza „ostatnie mecze jednej drużyny".
