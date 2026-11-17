# Prompt do Claude Code — runda 6: PDF dla zakładki „Perspektywy grupy"

> Rundy 1–5 zastosowane. Skopiuj wszystko poniżej linii.

---

Dodaj eksport PDF dla zakładki „Perspektywy grupy" w
`football_ai/streamlit_app.py`. Nie dotykaj `features/`, `training/`,
`models/`, `predictions/service.py`, `insights/` ani `simulation/`.

## Diagnoza — dlaczego dziś tego nie ma

`_render_pdf_export()` (ok. linii 3153) buduje PDF wyłącznie z
`st.session_state["selected_fixture"]` — stanu, który ustawia tylko zakładka
„Nadchodzące mecze". `_render_competition_outlook_tab()` trzyma swój wynik
pod kluczem `outlook_result` i nigdy nie jest sprawdzany przez eksport.
Przycisk w `_render_header()` (współdzielony nagłówek nad wszystkimi
zakładkami) po prostu nie ma czego pokazać, gdy `selected_fixture` jest puste
— nie jest zepsuty, tylko podłączony do jednego z dwóch źródeł danych.

**Nie przerabiaj wspólnego przycisku na „zgadywanie", która zakładka jest
aktywna** — Streamlit nie ma do tego czystego mechanizmu bez dodatkowego
stanu, i to nadmiernie komplikuje coś prostego. Zamiast tego: dodaj **drugi,
niezależny przycisk eksportu wewnątrz samej zakładki „Perspektywy grupy"**,
tak jak są tam już wydzielone sekcje (`st.container(border=True)` dla tabeli,
wykresu awansu i mapy pozycji) — eksport PDF jako kolejna taka sekcja, blisko
dołu, po scenariuszu „co jeśli".

## A. Nowa funkcja `_build_outlook_pdf(...)`

Wzoruj się bezpośrednio na `_build_analysis_pdf()` (ok. linii 1944) — ta sama
klasa `_BrandedPDF` (znak wodny na każdej stronie przez `footer()`), ten sam
font DejaVu, ten sam wzorzec `write()`/`embed_chart()`. Zawartość:

1. **Nagłówek**: „Dywizja X, grupa Y", sezon, data wygenerowania — ten sam
   wzorzec co „match/date/venue u góry" w istniejącym PDF (edytor musi to
   rozpoznać, zanim przeczyta cokolwiek dalej).
2. **Aktualna tabela grupy** — te same kolumny co `standings_frame` na
   ekranie, jako prawdziwa tabela PDF (`document.multi_cell` per wiersz albo
   wbudowana tabela fpdf2, tak jak robi to `render_form_table()` dla formy
   drużyn w istniejącym PDF — nie renderuj tabeli jako obrazka).
3. **Wykres prawdopodobieństwa awansu** — nowa funkcja
   `_pdf_probability_chart_png(labels, values)`, uogólnienie
   `_pdf_outcome_chart_png()` na N drużyn zamiast trzech stałych kategorii
   (te same poziome słupki, ta sama rodzina kolorów co wykres na ekranie —
   `_probability_chart`).
4. **Mapa cieplna pozycji końcowych** — nowa funkcja
   `_pdf_position_heatmap_png(...)`, ta sama logika co `_pdf_score_grid_png()`
   (ten sam colormap z `_METRIC_HEATMAP_COLORSCALE`, ten sam dobór koloru
   tekstu przez `_readable_cell_text_color`), tylko osie to drużyny × miejsca
   końcowe zamiast bramek gospodarza × gościa.
5. **Zastrzeżenie o tiebreaku** — to samo zdanie, które jest dziś pod tabelą
   na ekranie (`st.caption("⚠️ Uproszczenie: ...")`), jako zwykły tekst w PDF.
   Nie pomijaj go — to jest ten sam powód co w Rundzie 3: produkt sprzedający
   się na sprawdzalności nie może chować w PDF-ie tego, co pokazuje na ekranie.
6. **Scenariusz „co jeśli", jeśli został policzony w tej sesji** — sprawdź
   `st.session_state` pod kluczami, których używa `_render_what_if()`
   (`whatif_match`, `whatif_home_goals`, `whatif_away_goals`); jeśli obecne,
   dolicz tabelę „przed/po" jako ostatnią sekcję. Jeśli nieobecne, po prostu
   pomiń tę sekcję — nie generuj jej na siłę z domyślnymi wartościami, bo to
   nie jest scenariusz, o który realnie zapytano.

## B. Wpięcie przycisku w zakładce

Po `_render_what_if(...)` w `_render_competition_outlook_tab()`, w kolejnym
`st.container(border=True)`: przycisk `st.download_button("Eksportuj do
PDF", ...)`, ta sama nazwa co istniejący, żeby nie wprowadzać nowego
wzorca UI. Nazwa pliku: `f"{division}-{group}-{season}.pdf"` (oczyszczona ze
spacji tak jak dziś robi to eksport meczu). Ten sam wzorzec `try/except` →
`st.caption("PDF niedostępny")` w razie błędu, żeby jeden zepsuty wykres nie
wywalał całej zakładki.

## Weryfikacja

1. `ruff check .` i `pytest football_ai/tests` przechodzą.
2. Wygeneruj PDF dla dwóch różnych grup (dywizja z pełną tabelą i taka, gdzie
   drużyny mają zera meczów rozegranych) — druga nie może się wysypać na
   pustych danych.
3. Wygeneruj PDF raz bez uruchamiania scenariusza „co jeśli" i raz po jego
   uruchomieniu — sekcja scenariusza pojawia się tylko w drugim przypadku.
4. Sprawdź polskie znaki w nazwach drużyn w tabeli i na wykresach.
5. Potwierdź, że znak wodny jest na **każdej** stronie, jeśli dokument
   wyszedł dłuższy niż jedna (przy grupie z wieloma drużynami mapa pozycji
   może to spowodować).

## Czego nie robić

- Nie zgaduj aktywnej zakładki w `_render_pdf_export()` — osobny przycisk.
- Nie renderuj tabeli standings jako obrazka, skoro fpdf2 ma natywne tabele.
- Nie generuj sekcji „co jeśli" bez realnie uruchomionego scenariusza.
- Nie pomijaj zastrzeżenia o tiebreaku w wersji PDF.
