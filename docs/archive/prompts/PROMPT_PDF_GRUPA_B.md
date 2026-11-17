# Prompt do Claude Code — wygeneruj PDF dla Ligi Narodów, dywizja B grupa 4

> Skrypt już istnieje i jest gotowy: `football_ai/scripts/generate_outlook_pdf_manual.py`.
> Napisałem go i zweryfikowałem logikę wykresów (matplotlib) w izolacji, ale
> samego składania PDF przez fpdf2 nie mogłem uruchomić end-to-end w
> środowisku, w którym go pisałem (brak fpdf2/streamlit/sieci tam) — to jest
> pierwszy realny przebieg. Skopiuj wszystko poniżej linii.

---

Uruchom istniejący skrypt i zweryfikuj wynik. Nie pisz nowego kodu, chyba że
skrypt się wywali — wtedy zdiagnozuj i popraw najmniejszą możliwą zmianą, nie
przepisuj go od nowa.

To lekka operacja (fetch jednej strony Wikipedii + symulacja Monte Carlo +
złożenie PDF) — możesz to odpalić w osobnym terminalu równolegle z trwającym
treningiem, nie trzeba na niego czekać.

## A. Uruchomienie

Z katalogu głównego repo, w venv z pełnymi zależnościami:

```
python -m football_ai.scripts.generate_outlook_pdf_manual
```

Domyślnie: dywizja B, grupa 4 (realna grupa Polski, `season="2026–27"`), ze
scenariuszem „co jeśli" dla pierwszego nierozegranego meczu. Plik wyjściowy
wyląduje jako `B-4-2026–27.pdf` w katalogu, z którego odpalasz komendę —
**odpal z katalogu głównego repo**, żebym mógł go potem odczytać.

Odpal też wariant bez scenariusza „co jeśli" (V6 miało to jako osobny punkt
weryfikacji — sekcja ma się pojawiać tylko, gdy realnie policzona):

```
python -m football_ai.scripts.generate_outlook_pdf_manual --no-whatif --out B-4-2026-27-bez-whatif.pdf
```

## B. Jeśli się wywali

Najbardziej prawdopodobne miejsca:
- `UefaNationsLeagueSource.fetch_group_matches`/`find_team_group` — strona
  Wikipedii dla `"2026–27 UEFA Nations League B"` mogła się zmienić
  strukturalnie, albo grupa 4 już się skończyła/nie istnieje w obecnym
  sezonie (jeśli tak, spróbuj innej dywizji/grupy przez `--division`/`--group`
  albo `--team`).
- `PredictionService.create()` — jeśli akurat w trakcie treningu registry.json
  jest zapisywany, odczyt może trafić na niespójny stan; jeśli tak, po prostu
  spróbuj ponownie za chwilę, nie debuguj tego jako realny bug.
- Coś w `_build_outlook_pdf` samym (fpdf2) — to jest ta część, której nigdy
  nie uruchomiłem naprawdę, więc jeśli tu coś się posypie, to realna
  informacja zwrotna, zgłoś dokładny traceback zamiast próbować zgadywać
  poprawkę.

## C. Weryfikacja wyniku (z PROMPT_STREAMLIT_V6.md, nie pomijaj)

1. Otwórz oba wygenerowane PDF-y. Sprawdź: tabela grupy to prawdziwa tabela
   (da się zaznaczyć/skopiować tekst), nie obrazek.
2. Polskie znaki (ą/ę/ś/ż/ó...) w nazwach drużyn — w tabeli i na wykresach.
3. Zastrzeżenie o tiebreaku obecne jako tekst pod tabelą.
4. Wariant z „co jeśli" ma tę sekcję, wariant `--no-whatif` jej nie ma.
5. Jeśli dokument wyszedł dłuższy niż jedna strona (mapa pozycji przy grupie
   z wieloma drużynami może to spowodować) — znak wodny na **każdej** stronie,
   nie tylko pierwszej.
6. `ls -la *.pdf` i podaj mi rozmiar obu plików — chcę zobaczyć, że to nie
   puste/ucięte dokumenty, zanim je otworzę.

Zostaw oba pliki PDF w katalogu głównym repo (nie usuwaj po teście) — chcę
je stamtąd odczytać.
