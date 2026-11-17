# Prompt do Claude Code — sufit prawdopodobieństwa czy realny rezydualny przeciek?

> Diagnostyka, jak poprzednio. To NIE jest zaproszenie do zgadywania
> czwartego mechanizmu przecieku -- wyczerpałeś dwie konkretne hipotezy i
> słusznie się zatrzymałeś. To jeden dodatkowy pomiar (tani, częściowo już
> masz te dane) plus jeden test stabilności, żeby odróżnić "mały, legalny
> efekt matematyczny" od "szum" od "realny rezydualny bug", zanim
> podejmiemy decyzję (a) czy (b) z Twojego pytania. Skopiuj wszystko
> poniżej linii.

---

## Kontekst

Runda 4: Polska/Niemcy nadal ma odwrócony kierunek (0.068 > 0.037), ale
zauważ że różnica przy prawdziwym gospodarzu skurczyła się PROPORCJONALNIE
mocniej (-46%) niż na neutralnym gruncie (-20%) po usunięciu tożsamości
drużyny -- stąd ratio się pogorszyło (1.24x -> 1.85x) mimo że obie liczby
bezwzględnie spadły. To niekoniecznie oznacza trzeci przeciek. Hipoteza:
przy dużym rozjeździe Elo (Niemcy wyraźnie silniejsi) prawdziwa przewaga
gospodarza w kategoriach prawdopodobieństwa może być naturalnie mała --
prawdopodobieństwo jest ograniczone do [0,1], więc efekt bonusu za grę u
siebie ściska się bliżej górnej granicy, kiedy drużyna już ma wysokie
`home_win` z samej jakości. To by znaczyło, że punkt odniesienia ("różnica
przy prawdziwym gospodarzu") jest naturalnie mały właśnie dla tej pary --
nie dlatego, że coś przecieka, tylko dlatego, że sam miernik (ratio) traci
sens przy ekstremalnym Elo.

## 1. Surowe prawdopodobieństwa dla B1/B2 (Polska/Niemcy, prawdziwy gospodarz)

Zgłoś pełne `home_win`/`draw`/`away_win` dla B1 i B2 z rundy 4 (nie tylko
deltę `home_win - away_win`, którą już mamy). Jeśli masz je zapisane z
uruchomienia, nie odpalaj nic od nowa -- po prostu wypisz.

Jeśli `home_win` dla B2 (Niemcy prawdziwy gospodarz) jest już wysoko
(orientacyjnie >0.55-0.60) -- to popiera hipotezę sufitu: model już
przewiduje wyraźną dominację Niemiec z samej jakości, więc dodatkowy bonus
za grę u siebie ma mało miejsca, żeby się objawić jako duża delta.

Jeśli jest w okolicach 0.45-0.50 -- hipoteza sufitu upada, zostajemy z
realnym rezydualnym obciążeniem, nie efektem matematycznym.

Zrób to samo porównanie dla Portugalia/Holandia (D1/D2) jako punkt
odniesienia -- czy ich `home_win` jest wyraźnie niższy (dalej od sufitu)?
To by tłumaczyło, dlaczego ta para zachowuje się poprawnie konsekwentnie.

## 2. Test stabilności -- czy to w ogóle się powtarza

`--n-trials 10` (obniżone świadomie ze względu na koszt) i cokolwiek za
seedowanie losowości jest w `tuning.py`/`stacking_ensemble.py` -- sprawdź,
czy jest tam w ogóle jawny, ustawiany seed, czy nie. Jeśli nie ma jawnego
seeda, sam ponowny przebieg tej samej komendy z nowym tagiem wersji już
jest naturalnym testem "inny seed" -- nie trzeba nic dodatkowo zmieniać.
Odpal jeszcze raz retrening `match_outcome` (sam ten jeden target, tańsze
niż cała trójka) z nowym tagiem wersji, i porównaj: czy ratio dla
Polska/Niemcy zostaje w tej samej okolicy (~1.5-2x), czy skacze mocno
(np. raz 1.2x, raz 3x)? Stabilny wynik w podobnym rzędzie wielkości mimo
losowości treningu -- to realny sygnał, nie szum. Mocno skaczący wynik --
to szum przy 10 próbach Optuny na rzadkiej kombinacji danych, i wtedy
sensowniej podnieść `--n-trials` niż szukać czwartego mechanizmu.

## Co zgłosić z powrotem

1. Pełne prawdopodobieństwa B1/B2 i D1/D2 z rundy 4.
2. Czy jest jawny seed w tuning/stacking, czy nie.
3. Wynik drugiego przebiegu `match_outcome` -- ratio dla Polska/Niemcy,
   porównane z pierwszym.
4. Twoja zaktualizowana ocena: sufit prawdopodobieństwa, szum, czy dalej
   nie wiadomo.

## Czego nie robić

- Nie szukaj trzeciego mechanizmu przecieku na ślepo -- to wciąż
  obowiązuje.
- Nie trenuj drugiej i trzeciej tury (corners/cards/shots/possession) --
  to osobno, zablokowane przez ESPN/raport jakości danych, nie miesza się
  z tym.
