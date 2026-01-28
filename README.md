### Cel Projektu

Stworzenie modelu AI zdolnego do komponowania muzyki z możliwością sterowania nastrojem utworu w czasie rzeczywistym (w projekcie jest do tego parametr `danger_level`: Safe vs Battle, docelowo miałby to być stan w grze, np poziom życia gracza).

Utwory muzyczne (nie ma ich na repo) pochodzą z vgmusic, prawa należą do ich autorów.


### Przebieg prac
Między etapami prac modele były uruchamiane z zmienianymi parametrami nauki i tworzenia muzyki oraz inaczej przetworzonymi datasetami.
1. Trenowanie LSTM na muzyce z gry Final Fantasy 9 i zapisywanie muzyki na dysk
	1. Naiwne rozdzielanie utworów na Safe i Battle po gęstości nut
2. Trenowanie Transformera na muzyce z gry Final Fantasy 9 i zapisywanie muzyki na dysk
3. Dodanie do modelu nauki nie tylko dźwięku nuty ale też jej wartości rytmicznej
4. Stworzenie prostej aplikacji konsolowej do odtwarzania muzyki generowanej przez Transformer real time
5. Trenowanie Transformera na kilku, ręcznie wybranych utworach z Castlevani (Battle music) i Kingdom Hearts i Zelda: Ocarina of Time (Safe music) 
6. Trenowanie Transformera na pełnym zbiorze muzyki z FF9 (247 plików, nie więcej niż 10% z tego zostało odrzucone w preprocessingu)
	1. Polepszenie rozdzielania muzyki 



### Wybrane parametry
#### Preprocess
- step
	- co ile nut przesuwamy okno podczas tworzenia datasetu treningowego
	- Mały step (1-3) to dużo danych treningowych
	- Duży step (10-15) to mniej nakłądające się dane treningowe
	- Wielkość zależy od ilości danych treningowych (im mniej tym mniejszy step)
- danger_treshold
	- średni czas między nutami by uznać utwór za Battle music
	- Należy dostosować zależnie od zbioru utworów (eksperymentalnie, by średnio to co brzmi bojowo dostało odpowiednią etykietę)
- sequence_length
	- Wielkość pamięci, ile nut wstecz model widzi
	- Za mało (10) to brak motywów w utworze lub ogólny chaos
	- Za dużo (100) to wolny model i trudna zmiana nastroju w trakcie generowania 
	- Zazwyczaj używałem 30-50
   
#### Generowanie muzyki
- temperature
	- Niska temperatura (0.6) ogranicza kreatywność modelu, wybiera rzeczy o wysokim prawdopodobieństwie, łatwiej się zapętla
	- Wysoka temperatura (1.5) sprawia że ignoruje szkolenie i często wybiera dźwięki o niskim prawdopodobieństwie
	- Zazwyczaj używałem 0.8-1.0
	- Istnieje wariant temperatury dla dźwięku i wartości rytmicznej
- danger_level
	- Wybiera etykietę (Battle = 0.0 / Safe = 1.0) na której model powinien się skupić
	- Obecnie muzyka jest dzielona tylko na dwie kategorie, ale nie powinno być problemu z 
	- W skrypcie generującym cały utwór na raz, danger_level ustala się przed procesem
	- W muzyce czasu rzeczywistego można sterować danger_level za pośrednictwem konsoli



### Problemy i ich rozwiązania
#### Słownik
- Akordy
	- Aby nie tworzyć nieporęcznie dużego słownika dla AI, akordy (kilka nut naraz) miały w słowniku sortowane dźwięki. 
	- Aby jeszcze uprościć sprawę, ostatecznie brany jest tylko najwyższy dźwięk
- Tempo a nuty
	- to że nuta trwa krótko (np ćwierćnuta), jeszcze nie znaczy że należy do szybkiego utworu - bo tempo (BPM) może być niskie, np 60
- Wyizolowanie tracku melodii
	- We wczesnej fazie projektu, w wygenerowanej muzyce często pojawiały się niepasujące do reszty niskie dźwięki
	- W diagnozie problemu pomocne okazało się przetestowanie na skrypcie prostego i charakterystycznego utworu (Super Mario Bros)
	- Okazało się, że zapomniałem wyciąć basową ścieżkę z utworu przed użyciem metody `flatten`, która ustawiała po kolei jednocznesne ścieżki np melodię i bas, co uczyło model grać na zmianę linie melodyczne i basowe

#### Podział na Safe/Battle
- Licząc interwały na potrzebę oceny danger/safe trzeba to zrobić PRZED podziałem utworu na track melodii
	- W przeciwnym wypadku nie weźmiemy pod uwagę dźwięków, które nawet jeśli nie są główną linią, nadal wpływają na szybkość utworu
- Gęstość nut
	- Domyślnie po prostu liczyłem gęstość nut po dokonaniu `flatten` 
		- `flatten` spłaszcza wszystkie ścieżki utworu i akordy w jedną duża listę
	- Pracując na pełnym zbiorze FF9 poprawiłem by akordy i ścieżki grające w tym samym momencie były traktowane jako jeden dźwięk.
		- Bez tego jeden akord złożony z kilku nut sugerował o wiele bardziej dynamiczny utwór niż w rzeczywistości był.
	- Przy okazji, w razie ciszy na końcu utworu, jest ona teraz ignorowana

#### Nauka
- Czas między dźwiękami
	- Okazuje się, że `quarterLength`, czyli wartość rytmiczna z biblioteki music21 nie wystarczy by poprawnie ocenić czas trwania dźwięku - półnuta będzie trwać inny czas zależnie od ustalonego BPM (beats per minute), które może być różne w poszczególnych piosenkach datasetu

#### Odtwarzanie muzyki
- Zmiana battle/safe
	- dla stabilności kilka następnych nut jest zapisywanych w buforze, teoretycznie kosztem dynamicznej zmiany nastroju, ale zmiana nastroju jest ograniczona o wiele bardziej przez `SEQUENCE_LENGTH`
		- Ten parametr pozwala modelowi nauczyć się patternów w muzyce, ale utrudnia mu wyjście z obecnego nastroju
		- Przy `SEQUENCE_LENGTH = 50` częściej utykał w obecnym nastroju niż udawało mu się wydostać z lokalnego minimum
	- Rozwiązaniem jest zabronienie modelowi generowania szybkich nut przy Safe i wolnych nut przy Battle
		- Jest to drobne oszustwo, ale pozwala modelowi wyraźnie przełączać się między nastrojami, zachowując kontekst poprzedniego nastroju. 
#### Inne
- Niskie dźwięki mają większą tendencję do nieskończonych pętli niż wysokie
- W rzadkich sytuacjach, przetwarzając utwory, skrypt dostawał z pliku MIDI coś co nie było ani akordem ani nutą. 


### Wnioski
#### LSTM vs Transformer
- Czas uczenia
	- LSTM był szybszy
		- Jedna epoka pełnego datasetu LSTM - ok. 1 minuty 40 sekund
		- Jedna epoka pełnego datasetu Transformer - ok. 5 minuty 50 sekund
	- Do treningu używałem CPU. Sieci LSTM przetwarzają sekwencyjnie, ale Transformer równolegle, więc mogłyby przetwarzać się o wiele szybciej na odpowiedniej karcie graficznej
- Oba modele mają tendencję do zacinania się na jednym, nieskończonym dźwięku
- Trudno jest porównać jakość muzyki, ale powiedziałbym że transformer potrafi tworzyć bardziej spójne melodie

#### Preprocessing
- Dobranie danych treningowych, oznakowanie go, odfiltrowanie chaotycznych/jednostajnych utworów, przycięcie nieodpowiednich tracków, wybranie które części składowe muzyki trafią do datasetu...
- Jest tu bardzo dużo czynników, o które można się potknąć jeszcze przed przystąpieniem do trenowania samego modelu.
- Możliwe, że mimo wielu poprawek nadal pominąłem jakiś istotny element
- Osoba z wykształceniem muzycznym powinna nadzorować proces preprocessingu
  
#### Generowanie muzyki
- Model jest jak najbardziej w stanie generować muzykę na bieżąco ją grając, nuta po nucie.
- Jakość muzyki w tej pracy pozostawia dużo do życzenia, ale dataset z FF9 może nie być najlepszy do tego zastosowania
	- W docelowej aplikacji można rozważyć skomponowanie własnej muzyki z myślą konkretnie o uczeniu maszynowym
	- Potrzebne by też były zabezpieczenia przed powtarzaniem tej samej nuty oraz cyklu nut (np Do Re Mi Fa Sol...)
- Przełączanie się między nastrojami Safe/Battle
	- Model zazwyczaj wybiera bardziej różnorodne (wysokie i niskie) dźwięki dla Battle music i bardziej spójny dla Safe music
	- Dodatkowa ingerencja (zablokowanie niektórych wartości rytmicznych) była wskazana, aby ułatwić modelowi zrozumienie konieczności przełączenia na inne neurony
	- "Ręczna" zmiana instrumentu pomaga użytkownikowi w rozróżnieniu nastrojów
		- Nauka który instrument gra z jaką melodią i wartością rytmiczną prawdopodobnie stworzyłaby chaos, z nielogicznym przełączeniem instrumentów
	- Mimo to, można ten punkt uznać za sukces, ponieważ generowana muzyka płynnie zmienia nastrój, zachowując kontekst poprzednich nut



### Kolejność uruchomiania
I.  preprocess.py
- ustalić ścieżkę z plikami midi w `folders_config` (albo wszystkie utwory w jednym do automatycznego podziału, albo osobne foldery dla calm i battle music)
- eksperymentalnie ustalić danger_treshold do podziału muzyki

II.  train.py -> dla LSTM
II.  transfortmer_train.py -> dla Transformera
- ustalić `DATA_PATH` z danymi w formacie `.pkl`

III. predict.py -> dla LSTM
III.  transformer_predict.py -> dla Transformera
IIIa.  real_time_music.py -> dla Transformera
- ustalić `DATA_PATH` z danymi w formacie `.pkl`
- ustalić `MODEL_PATH` z danymi w formacie `.pkh` (Transformer) kub `.keras` (LSTM)



