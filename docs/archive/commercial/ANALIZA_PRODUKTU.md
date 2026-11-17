# Analiza produktu: football_ai

**Data:** 29 lipca 2026
**Kontekst:** ocena obecnego kierunku projektu na tle raportu „Podsumowanie raportu" (8 pomysłów)
oraz realnego stanu kodu w repozytorium.
**Charakter dokumentu:** krytyczny. Celowo szukam słabych punktów, nie potwierdzam założeń.

> Analiza konkurencji ma krótki termin przydatności. Sekcję „Krajobraz konkurencyjny"
> warto odświeżyć za 2–3 miesiące.

---

## 1. Werdykt w jednym zdaniu

**Nie jesteśmy na pograniczu pomysłu 5 i 7 — jesteśmy *przed* nimi oboma, budując silnik,
którego każdy z nich by potrzebował, ale sam silnik nie jest produktem, a obecne
pozycjonowanie („B2B API z predykcjami") to towar masowy z darmowym, lepszym substytutem.**

---

## 2. Gdzie naprawdę jesteśmy

Zestawienie deklarowanego kierunku z tym, co realnie jest w kodzie:

| Element | Pomysł 5 (media/insighty) | Pomysł 7 (symulacje) | Stan w repo |
|---|---|---|---|
| Baza meczów historycznych | rdzeń produktu | wsad do kalibracji | Częściowo — 25k meczów, ale tylko 2000–2026 |
| Rekordy, serie, kamienie milowe | rdzeń produktu | — | **Brak kodu** |
| Detekcja anomalii / „co jest ciekawe" | rdzeń produktu | — | **Brak kodu** |
| Generowanie tekstu (NLG/LLM) | warstwa prezentacji | — | **Brak kodu** |
| Symulacje Monte Carlo | — | rdzeń produktu | **Brak kodu** |
| Scenariusze „co jeśli" | — | rdzeń produktu | **Brak kodu** |
| Prawdopodobieństwa 1X2 | wsad | wsad | Gotowe |
| Predykcje niszowe (rożne, kartki, strzały) | wsad | — | Gotowe |
| Elo / forma / H2H | wsad | wsad | Gotowe |
| Wyjaśnialność (SHAP) | wartość dodana | — | Infrastruktura jest, produktu nie ma |

**Realistyczna ocena pokrycia: ~5% pomysłu 5, ~15% pomysłu 7.**

To, co mamy, to solidna infrastruktura predykcyjna — czyli *wsad* do obu pomysłów.
Sam wsad nikt nie kupi.

**Największe ryzyko na dziś nie jest techniczne, tylko strategiczne:** można w nieskończoność
ulepszać silnik (więcej cech, GNN, lepszy tuning) i nigdy nie zbudować produktu.
Projekt ma dziś 842 cechy i zero funkcji, za którą ktokolwiek zapłaciłby.

---

## 3. Twardy problem: predykcje 1X2 to towar masowy

To jest najważniejsza sekcja tego dokumentu.

**Fakty rynkowe:**

- Dokładność bukmacherów dla piłki nożnej to ~54% przy 1X2. To jest realny sufit rynku, nie nasz.
- Kursy bukmacherskie **przewyższają** metody oparte na Elo, średniej punktów i bramkach —
  zweryfikowane na ~100 tys. meczów. Nasz obecny rdzeń cech (Elo + forma + H2H) to dokładnie
  ta rodzina metod, która w tym badaniu przegrała z kursami.
- Najlepsze modele akademickie *prawie dorównują* rynkowi (70,2% vs 70,6% u Betfair) — czyli
  po ogromnym wysiłku osiąga się wynik nieco gorszy niż darmowa baseline.

**Co to znaczy dla modelu biznesowego:**

Klient B2B, któremu chcemy sprzedać prawdopodobieństwa, może wziąć implikowane
prawdopodobieństwa z dowolnego API z kursami — taniej, natychmiast i **lepiej skalibrowane
niż nasz model**. Sprzedajemy produkt, którego lepsza wersja jest darmowa.

To jest najgorsza możliwa kombinacja: twardy sufit dokładności + darmowy, lepszy substytut
+ nasycony rynek dostawców.

**Wniosek:** „nasze liczby są lepsze" nie może być osią produktu. Nigdy nie będą.

---

## 4. Krajobraz konkurencyjny

### 4.1 Kto realnie zajmuje pomysł 5 (raport tego nie wychwycił)

Raport wymienił jako konkurencję Wyscout, InStat, Opta, StatsBomb, SkillCorner — czyli
dostawców **danych i scoutingu**. To pominięcie: bezpośrednia konkurencja dla pomysłu 5 to
dwie inne firmy, które robią dokładnie to, co opisuje pomysł 5, i robią to od lat.

**Gracenote (Nielsen)** — ma usługę *Facts & Figures*: serie zwycięstw i porażek, rekordy,
kamienie milowe, bilanse H2H, historia rozgrywek, statystyki przedmeczowe — dostarczane
mediom przed każdym wydarzeniem. Do tego *SportsDesk Live* — zespół redakcyjny dostarczający
dziennikarzom „unikalne kąty" i gotowe informacje, dla szerokiego zakresu lig piłkarskich.
Pokrycie: 4500+ lig i wydarzeń.

**To jest pomysł 5, wdrożony, sprzedawany, z zespołem redakcyjnym w środku.**

**Stats Perform (Opta)** — *Automated Insights*: generatywna AI zamieniająca bazę Opty
w gotowe narracje, „od długich raportów meczowych, przez kamienie milowe na żywo, po
krótkie fakty", w wielu językach. Do tego 140+ modeli AI i 8 modeli fundamentowych
obejmujących predykcje, **symulacje**, analizę i storytelling. W maju 2026 dorzucili
Opta Pulse (automatyczne klipy wideo).

**Stats Perform pokrywa jednocześnie pomysł 5 i pomysł 7.**

**United Robots** — automatyczne dziennikarstwo sportowe jako „przedłużenie działu
sportowego" redakcji, z naciskiem na lokalne rozgrywki. Dowód, że model biznesowy
„automatyczne treści dla mediów lokalnych" działa komercyjnie.

### 4.2 Macierz zdolności

Ocena uczciwa — tam, gdzie przegrywamy, jest napisane, że przegrywamy.

| Obszar | My (dziś) | Stats Perform / Opta | Gracenote | Darmowe API kursów | EkstraStats.pl |
|---|---|---|---|---|---|
| **Dane** | | | | | |
| Głębokość historyczna | Słaba (2000+) | Bardzo mocna | Bardzo mocna | Brak | Adekwatna |
| Pokrycie lig | Słabe | Bardzo mocne | Bardzo mocne (4500+) | Mocne | Tylko Ekstraklasa |
| Dane zdarzeniowe (event data) | Brak | Bardzo mocne | Mocne | Brak | Słabe |
| Licencja na dane Ekstraklasy | **Brak** | Prawdopodobnie tak | Prawdopodobnie tak | — | Niejasne |
| **Predykcje** | | | | | |
| 1X2 | Adekwatne | Mocne | Adekwatne | **Mocne** (lepiej skalibrowane) | Brak |
| Rynki niszowe (rożne, kartki) | **Mocne** | Mocne | Słabe | Słabe | Brak |
| Symulacje Monte Carlo | Brak | Mocne | Adekwatne | Brak | Brak |
| **Treść redakcyjna** | | | | | |
| Rekordy / serie / ciekawostki | **Brak** | Mocne | **Bardzo mocne** | Brak | Adekwatne |
| Generowanie narracji | Brak | Mocne | Mocne | Brak | Brak |
| Weryfikowalność faktów | Brak | Adekwatna | Mocna (zespół redakcyjny) | — | Adekwatna |
| **Pozycja** | | | | | |
| Wyjaśnialność jako produkt | **Adekwatna** (SHAP) | Słaba | Słaba | Brak | Brak |
| Język polski / kontekst lokalny | **Mocny** (potencjał) | Adekwatny | Adekwatny | Słaby | **Mocny** |
| Cena dla małego wydawcy | **Mocna** (potencjał) | Słaba (enterprise) | Słaba (enterprise) | Mocna | — |
| Niższe ligi / młodzież | Brak | Słabe | Słabe | Słabe | Brak |

### 4.3 Co z tej macierzy wynika

Przegrywamy w **każdej kategorii związanej z danymi i skalą** — i nigdy tego nie odrobimy.
Opta i Gracenote mają dekady zbierania danych, zespoły redakcyjne i umowy licencyjne z ligami.

Wygrywamy (albo możemy wygrać) w **czterech wąskich miejscach**:

1. **Rynki niszowe** — rożne, kartki, strzały. Mniej płynne rynki, mniej efektywne kursy,
   mniej konkurentów je modeluje. Już to mamy.
2. **Wyjaśnialność jako produkt** — nikt nie sprzedaje „dlaczego model tak uważa" jako
   głównej wartości. Raport sam wskazuje XAI jako najbardziej obiecujący kierunek.
   Mamy infrastrukturę SHAP i bogatą przestrzeń cech.
3. **Język polski i kontekst lokalny** — Opta/Gracenote są wielojęzyczne, ale ich głębia
   redakcyjna dla Ekstraklasy i polskiego kontekstu jest prawdopodobnie płytka.
4. **Punkt cenowy** — mały polski wydawca nie kupi enterprise'owego kontraktu z Nielsenem.

---

## 5. Trzy drastyczne propozycje

### 5.1 Porzucić pozycjonowanie „uniwersalne API predykcyjne"

Uzasadnienie w sekcji 3: towar masowy, twardy sufit, darmowy lepszy substytut.
Silnik zostaje — zmienia się to, co sprzedajemy. Predykcja przestaje być produktem,
a staje się **składnikiem**.

Konsekwencja: lista docelowych klientów z opisu projektu („portale, media, startupy,
aplikacje mobilne, firmy analityczne, platformy informacji bukmacherskiej") jest za szeroka
i wewnętrznie sprzeczna. Platforma bukmacherska i redakcja sportowa chcą zupełnie innych
rzeczy. Trzeba wybrać jedną.

### 5.2 Cofnąć decyzję o top 5 lig + Lidze Mistrzów

To bezpośrednio przeczy temu, co ustaliliśmy godzinę wcześniej — ale zostałem poproszony
o krytykę nawet drastyczną, więc: **top 5 lig to najgorszy możliwy wybór dla produktu
redakcyjnego.**

Każda statystyka o Realu Madryt jest już policzona przez 50 firm i rozdawana za darmo.
Dziennikarz piszący o Premier League ma Optę, FBref, Transfermarkt, WhoScored i dziesiątki
darmowych źródeł. Nasza wartość dodana ≈ 0.

Raport sam wskazuje lukę i jest to luka gdzie indziej: *„brak zaawansowanego systemu
raportowania dla niższych lig czy młodzieży"*.

**Uczciwe zastrzeżenie (to nie jest czysty wybór):** popyt polskich mediów jest większy
na top 5 lig niż na I ligę, a dane top 5 są łatwo dostępne (FBref, Football-Data.co.uk).
Natomiast dane Ekstraklasy są **prawnie zamknięte** — Ekstraklasa SA jest właścicielem
statystyk i komercyjne wykorzystanie wymaga pisemnej zgody. To realna bariera wejścia,
która działa w obie strony: blokuje nas teraz, ale byłaby fosą, gdybyśmy ją przeszli.

**Rekomendacja:** top 5 lig owszem, ale jako *kontekst i tło*, nie jako rdzeń oferty.
Rdzeń powinien być tam, gdzie nikt nie patrzy.

### 5.3 Odwrócić wartość: weryfikacja, nie generowanie tekstu

Raport opisuje w pomyśle 5 „narzędzia NLP/LLM do formatowania wniosków w czytelny tekst"
jako kluczową technologię. **To jest odwrócenie wartości w erze LLM-ów.**

Generowanie ładnych zdań jest dziś darmowe — każdy dziennikarz ma ChatGPT. Jeśli naszą
wartością jest „piszemy zgrabne zdania", zastępuje nas dowolny model językowy w 5 sekund.

Wartością jest **gwarancja, że fakt jest prawdziwy i ma źródło**. „To będzie pierwszy taki
wynik od 1962 roku" — jeśli to nieprawda, dziennikarz publikuje błąd rzeczowy pod swoim
nazwiskiem. Odpowiedzialność jest ogromna, a zaufanie jest jedyną rzeczą, której LLM
nie skopiuje.

Konsekwencja: warstwa NLG jest ostatnia i najmniej ważna. Warstwa weryfikacji faktów
(deterministyczna, testowalna, z odnośnikiem do konkretnych meczów w bazie) jest rdzeniem.

---

## 6. Rekomendowane pozycjonowanie

Zamiast „B2B API do predykcji meczów piłkarskich":

> Dla **polskich redakcji sportowych i komentatorów**, którzy przed każdym meczem muszą
> znaleźć ciekawe, prawdziwe i nieoczywiste fakty pod presją czasu, **football_ai** jest
> **automatycznym researcherem redakcyjnym**, który dostarcza zweryfikowany pakiet faktów
> i wyjaśnionych prognoz. W przeciwieństwie do Opty i Gracenote — jest w cenie osiągalnej
> dla średniego wydawcy, po polsku, i pokazuje *dlaczego*, a nie tylko *ile*.

### Jak łączą się pomysły 5 i 7

Intuicja o „pograniczu 5 i 7" jest dobra, ale relacja powinna być jawna i hierarchiczna:

**Pomysł 7 to fabryka treści. Pomysł 5 to produkt.**

Symulacje Monte Carlo generują fakty, których **nie ma nigdzie indziej**:

- „Polska ma 23% szans na awans z grupy"
- „Gdyby Lewandowski nie zagrał, szanse spadają do 17%"
- „Po tej porażce awans wymaga wygranej w dwóch ostatnich meczach — prawdopodobieństwo 31%"

To są zdania cytowalne, nagłówkowe i **niedostępne w żadnym darmowym źródle**, bo wymagają
modelu. Symulacja nie jest osobnym produktem (raport słusznie zauważa: *„mało efektowna
zabawa, raczej funkcja analityczna"*) — jest generatorem unikalnego contentu wewnątrz
produktu redakcyjnego.

Dodatkowo: symulacja jest tania do zbudowania, bo silnik prawdopodobieństw **już mamy**.

---

## 7. Co to zmienia w kodzie i roadmapie

| Obszar | Teraz | Po zmianie |
|---|---|---|
| Metryka sukcesu | log-loss / F1 na 1X2 | liczba faktów na mecz: prawdziwych × nieoczywistych × publikowalnych |
| Priorytet danych | więcej cech do modelu | pełna historia (1872) + weryfikowalność |
| Warstwa docelowa | `/predict` | `/matches/{id}/preview`, `/recap`, moduł `insights/` |
| Rola modelu ML | produkt | składnik generujący fakty |
| Rola SHAP | debugowanie modelu | **funkcja produktowa** (uzasadnienie prognozy) |
| Testy | czy kod działa | czy konkretny fakt historyczny jest poprawny (golden cases) |
| Symulacje | brak | Monte Carlo nad istniejącym silnikiem |

Uwaga: naprawa przecieku danych i chronologicznej walidacji (zrobiona wcześniej) pozostaje
w pełni zasadna — silnik musi być poprawny niezależnie od tego, co sprzedajemy.
Ale **kolejne miesiące na dostrajanie modelu nie zbliżają do produktu**.

---

## 8. Najbardziej ryzykowne założenie

**Że polskie redakcje sportowe za to zapłacą.**

To jest całkowicie niezweryfikowane. Nie rozmawialiśmy z ani jednym dziennikarzem.
Możliwe kontrargumenty, których nie znamy:
- redakcje mają już abonament u dostawcy danych i to im wystarcza,
- research robi stażysta za darmo i nikt nie widzi problemu,
- budżety są tak małe, że nawet niska cena jest za wysoka,
- decyzja zakupowa trwa 9 miesięcy i przechodzi przez centralę wydawnictwa.

**Następny krok — jedyny, który ma teraz sens:** rozmowa z 5 polskimi dziennikarzami
sportowymi lub redaktorami. Nie demo, nie prezentacja — wywiad o tym, jak dziś przygotowują
się do meczu, ile to zajmuje i co ich najbardziej wkurza. Zanim powstanie kolejna linijka kodu.

Pytania, które warto zadać:
1. Jak wygląda Twoje przygotowanie do relacji/artykułu przedmeczowego? Ile czasu zajmuje?
2. Skąd bierzesz statystyki i ciekawostki? Co jest w tym najbardziej upierdliwe?
3. Kiedy ostatnio nie znalazłeś czegoś, czego szukałeś?
4. Czy zdarzyło Ci się opublikować statystykę, co do której nie miałeś pewności?
5. Kto w Twojej redakcji decyduje o zakupie narzędzi i jaki jest realny budżet?

---

## 9. Co porzucić (świadomie)

- **GNN** — raport wymienia to jako pomysł 2 (osobny produkt dla trenerów, wysoka złożoność).
  W naszym kontekście to badawcza ciekawostka bez ścieżki do przychodu. Zostawić jako
  szkielet, nie inwestować.
- **Predykcje jako główny produkt** — patrz sekcja 3.
- **Szeroka lista klientów** — wybrać jeden segment (redakcje) i zignorować resztę do czasu
  pierwszych 3 płacących klientów.
- **Warstwa NLG na początku** — najpierw zweryfikowane fakty w JSON, dopiero potem ładne zdania.
- **Rozszerzanie na kolejne ligi „bo można"** — każda nowa liga to koszt utrzymania,
  a nie automatycznie nowa wartość.

---

## 10. Zestawienie: pomysł 5 vs 7 vs my

| Kryterium | Pomysł 5 (media) | Pomysł 7 (symulacje) | Rekomendacja |
|---|---|---|---|
| Potencjał wg raportu | Wysoki | Średni | — |
| Realna konkurencja | **Bardzo wysoka** (Gracenote, Stats Perform) | Wysoka (Stats Perform) | — |
| Możliwe do zbudowania w pojedynkę | Tak | Tak | — |
| Nasze obecne pokrycie | ~5% | ~15% | — |
| Wymaga nowych danych | Tak (pełna historia) | Nie | — |
| Unikalność treści wyjściowej | Średnia | **Wysoka** | — |
| **Rola docelowa** | **Produkt** | **Silnik treści wewnątrz produktu** | Połączyć hierarchicznie |

---

## Źródła

- [Gracenote — Sports Data / Facts & Figures](https://gracenote.com/products/sports-data/)
- [Gracenote Mediaservices — dokumentacja](http://sportsdemo1.gracenote.com/documentation/mediaservices/all-sports)
- [Stats Perform — Automated Insights](https://www.statsperform.com/automated-insights/)
- [Stats Perform — AI w sporcie (140+ modeli)](https://www.statsperform.com/artificial-intelligence-in-sport/)
- [Stats Perform — premiera Opta Pulse, maj 2026](https://www.statsperform.com/resource/opta-pulse-launch/)
- [United Robots — automatyczne dziennikarstwo sportowe](https://www.unitedrobots.ai/content-services/sports)
- [Ekstraklasa SA — statystyki i warunki wykorzystania](https://www.ekstraklasa.org/statystyki/)
- [Badanie efektywności rynku bukmacherskiego (arXiv)](https://arxiv.org/pdf/2003.09384)
- [ML a prognozowanie wyników i kursów bukmacherskich (arXiv)](https://arxiv.org/pdf/2403.16282)
- [Łączenie ML i ekspertów — model bazowy (arXiv)](https://arxiv.org/pdf/2012.04380)
