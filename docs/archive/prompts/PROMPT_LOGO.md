# Prompt do Claude Code — wpięcie logo StatsDesk

> Logo jest już zaprojektowane i leży w `football_ai/assets/branding/`:
> `statsdesk_mark.svg` (sam znak), `statsdesk_lockup_dark.svg` (znak + nazwa,
> na ciemne tło), `statsdesk_lockup_light.svg` (znak + nazwa, na jasne tło/PDF),
> `statsdesk_watermark_dark.svg` i `statsdesk_watermark_light.svg` (przytłumiony
> znak wodny, dwie wersje pod jasne/ciemne tło). Nie zmieniaj tych plików —
> traktuj je jako gotowy zasób.

---

Wepnij logo StatsDesk w `football_ai/streamlit_app.py` w trzech miejscach.
Zastąp nim placeholder „TRAFF STATS".

## 1. Rasteryzacja — zrób raz, nie w runtime

`fpdf2` (którego już używacie do eksportu) nie ma pewnego wsparcia dla SVG we
wszystkich wersjach — rasteryzuj każdy plik do PNG w 3x rozdzielczości (logo
się nie zmienia, więc rób to raz jako krok budowy, nie przy każdym eksporcie).
Użyj `cairosvg` (dodaj do `requirements-local.txt`/`pyproject.toml` jako
zależność deweloperską) albo jednorazowego skryptu, który zapisze PNG obok
źródłowych SVG w tym samym katalogu. Do samej aplikacji (`st.image`,
Plotly `layout.images`) SVG i tak nie jest potrzebne — Streamlit i Plotly
przyjmują PNG bez kłopotu.

## 2. Nagłówek strony

W miejscu obecnego tytułu „TRAFF STATS" wstaw `statsdesk_lockup_dark.png`,
wyśrodkowany, wysokość ok. 48–56px. Zostaw pod nim istniejący opis. Ustaw też
`st.set_page_config(page_icon=...)` na rastrową wersję `statsdesk_mark.svg`
(sam znak, mniejsza rozdzielczość, np. 64x64, wystarczy).

## 3. Znak wodny pod każdym wykresem i mapą cieplną

Dotyczy wszystkich wykresów Plotly (prognoza 1X2, forma, siatka wyników,
mapa pozycji, profil minutowy) i tabel/heatmap renderowanych przez
`st.dataframe`/`st.plotly_chart` w całej aplikacji.

Dla wykresów Plotly: dodaj `statsdesk_watermark_dark.png` jako
`layout.images` z `xref="paper"`, `yref="paper"`, zakotwiczony w prawym dolnym
rogu (`x=1, y=0, xanchor="right", yanchor="bottom"`), `sizex`/`sizey` małe
(np. 0.12 papieru), `opacity` zostaw tak jak w źródłowym pliku (już wbudowana
w PNG, nie dubluj przez dodatkowy parametr). Napisz jedną pomocniczą funkcję
`_add_watermark(figure)` i przepuść przez nią każdą figurę tuż przed
`st.plotly_chart` — nie kopiuj tego kodu do każdego wykresu osobno.

Dla tabel/heatmap bez natywnej obsługi warstw (np. `st.dataframe`) — owiń
w kontener z `st.container()` i umieść mały `st.image(watermark, width=90)`
w rogu przez kolumny (`st.columns` z pustą szeroką kolumną + wąską na obrazek)
albo zaakceptuj, że tam znak wodny idzie pod tabelą, nie na niej — ważne, żeby
był widoczny przy zrzucie ekranu całej sekcji, nie żeby był piksel w piksel
w rogu każdej komórki.

## 4. PDF (`_build_analysis_pdf`, fpdf2)

- Nagłówek strony: `statsdesk_lockup_light.png` zamiast/obok obecnego tekstu
  z nazwą — `document.image(path, x=..., y=10, w=60)` na górze pierwszej strony.
- Stopka: `statsdesk_watermark_light.png`, mały, w rogu każdej strony PDF
  (fpdf2: ustaw przez `set_y()` blisko dołu strony przed `image()`, albo hook
  na `footer()` jeśli już z niego korzystacie). To jest właściwe zabezpieczenie
  przed kradzieżą, o które prosił użytkownik — musi być na **każdej** stronie
  dokumentu, nie tylko pierwszej.

## Weryfikacja

1. `ruff check .` przechodzi.
2. Nagłówek aplikacji pokazuje logo, nie tekst „TRAFF STATS".
3. Wygeneruj PDF i sprawdź: logo w nagłówku, znak wodny widoczny w rogu każdej
   strony (jeśli dokument ma więcej niż jedną stronę).
4. Zrzut ekranu dowolnego wykresu/mapy cieplnej pokazuje znak wodny w rogu,
   czytelny, ale niekonkurujący z danymi.
5. Sprawdź na jasnym i ciemnym renderze (Streamlit dark theme vs. biała
   strona PDF), że użyta jest właściwa wersja (`_dark`/`_light`) — pomylenie
   ich da nieczytelny, prawie niewidoczny znak.
