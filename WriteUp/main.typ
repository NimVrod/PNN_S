#import "@preview/fletcher:0.5.8" as fletcher: diagram, edge, node

#set page(paper: "a4", margin: 2cm, footer: context [
  #grid(
    columns: (1fr, 1fr, 1fr),
    // Creates 3 equal zones
    align: (left, center, right),
    // Aligns content within those zones

    [],
    // 1. EMPTY LEFT (Crucial for balancing)

    // 2. CENTER Page Number
    smallcaps[#context counter(page).display()],

    // 3. RIGHT Location & Date
    smallcaps[Rzeszów #datetime.today().display()],
  )
])

#set heading(numbering: "1.1.1.", supplement: [Rozdział])
/* Whether to show heading as Rozdział, Podrozdział or Pod-podrozdział depends on the level of heading. For example, if we want to show heading with level 2 as Rozdział, level 3 as Podrozdział and level 4 as Podpodrozdział, we can use the following code:
#show heading.where(level: 2): set heading(supplement: [Podrozdział])
#show heading.where(level: 3): set heading(supplement: [Podrozdział])
*/

#set text(lang: "pl")
#set math.equation(numbering: "(1.1)")
#show figure: set block(breakable: true)
#show figure.where(kind: raw): set figure(supplement: [Listing])
//#show table: set block(breakable: false)

#let appendix(body) = {
  set heading(numbering: none, supplement: [Appendix])
  body
}

//Title page
#page(numbering: none, header: none, footer: none)[
  #align(top)[
    #grid(
      columns: (1fr, 1fr),
      rows: auto,
      align(left)[#image("Assets/rut.png", height: 2.5cm)], align(right)[#image("Assets/weii.png", height: 2.5cm)],
    )
  ]

  #align(center)[
    #text(weight: "extrabold", size: 1cm)[Sztuczna Inteligencja] //Subject

    #text(size: 0.5cm)[Projekt] // Laboratory number

    #text(size: 0.5cm)[Probabilistyczna sieć neuronowa] // Laboratory subject

    #align(bottom)[
      Hubert Dec -- 179474 -- L04 //Author information
      #linebreak()
      #datetime.today().display()
    ]

  ]
]

#outline(depth: 2)
#pagebreak()

= Opis problemu
Zadaniem projektu jest implementacja i analiza probabilistycznej sieci neuronowej (PNN) #cite(<specht1990pnn>) w kontekście klasyfikacji wiadomości e-mail jako spam lub nie-spam. Na podstawie zbioru danych spambase #cite(<spambase>). Zostaną również przeprowadzone eksperymenty badające wpływ różnych funkcji jądra i wartości parametru wygładzania $sigma$ na dokładność klasyfikacji.

//TODO: może to lepiej napisać niż haiku
Spam stanowi poważny problem w komunikacji internetowej, powodując obciążenia dla serwerów poczty i negatywne doświadczenie użytkowników. Klasyfikacja wiadomości e-mail jako spam lub nie-spam wymaga rozróżnienia na podstawie charakterystyk takich jak zawartość tekstu, formatowanie i metadane.

Probabilistyczne sieci neuronowe są szczególnie przydatne dla tego problemu ze względu na: brak iteracyjnego uczenia (szybkie trenowanie na zbiorze uczącym), bezpośrednią interpretację decyzji, odporność na szum i zmienność spamu, możliwość wyboru różnych funkcji jądra, oraz asymptotyczną zbieżność do optymalnego klasyfikatora bayesowskiego. Zbiór danych spambase zawiera 4601 wiadomości (57% spamu) -- wystarczającą wielkość dla efektywnego trenowania PNN.


= Część teoretyczna <teoria>

Probabilistyczna sieć neuronowa (PNN) #cite(<specht1990pnn>) jest siecią neuronową,
w której wyjście każdego neuronu interpretowane jest jako prawdopodobieństwo przynależności
do danej klasy. PNN opiera się na teorii estymacji gęstości jądrowej (ang. *Kernel Density
Estimation*, KDE) i jest szeroko stosowana w zadaniach klasyfikacji nadzorowanej.

== Model probabilistycznej sieci neuronowej
#figure(
  diagram(
    node-stroke: 1pt,
    edge-stroke: 1pt,
    node-fill: white,
    spacing: (1.5cm, 1cm),

    // Etykiety warstw
    node((0, -1), [*Input Layer* \ *Warstwa wejściowa*], stroke: none),
    node((1, -1), [*Pattern Layer* \ *Warstwa wzorców*], stroke: none),
    node((2, -1), [*Summation Layer* \ *Warstwa sumowania*], stroke: none),
    node((3, -1), [*Output Layer* \ *Warstwa wyjściowa*], stroke: none),

    // Neurony wejściowe
    node((0, 0.5), [$x_1$], shape: "circle", name: <i1>, width: 2.5em),
    node((0, 2.5), [$x_2$], shape: "circle", name: <i2>, width: 2.5em),

    // Neurony wzorcowe -- klasa A
    node((1, 0), [$P_("A1")$], shape: "circle", name: <pa1>, width: 2.5em),
    node((1, 1), [$P_("A2")$], shape: "circle", name: <pa2>, width: 2.5em),

    // Neurony wzorcowe -- klasa B
    node((1, 2), [$P_("B1")$], shape: "circle", name: <pb1>, width: 2.5em),
    node((1, 3), [$P_("B2")$], shape: "circle", name: <pb2>, width: 2.5em),

    // Neurony sumujące
    node((2, 0.5), [$Sigma_A$], shape: "circle", name: <sa>, width: 2.5em),
    node((2, 2.5), [$Sigma_B$], shape: "circle", name: <sb>, width: 2.5em),

    // Neuron wyjściowy
    node((3, 1.5), [Decision \ (Argmax)], shape: "circle", name: <out>, width: 5em),

    // Połączenia: wejście → wzorce
    edge(<i1>, <pa1>, "->", stroke: gray),
    edge(<i1>, <pa2>, "->", stroke: gray),
    edge(<i1>, <pb1>, "->", stroke: gray),
    edge(<i1>, <pb2>, "->", stroke: gray),

    edge(<i2>, <pa1>, "->", stroke: gray),
    edge(<i2>, <pa2>, "->", stroke: gray),
    edge(<i2>, <pb1>, "->", stroke: gray),
    edge(<i2>, <pb2>, "->", stroke: gray),

    // Połączenia: wzorce → sumatory (tylko w obrębie tej samej klasy)
    edge(<pa1>, <sa>, "->"),
    edge(<pa2>, <sa>, "->"),
    edge(<pb1>, <sb>, "->"),
    edge(<pb2>, <sb>, "->"),

    // Połączenia: sumatory → wyjście
    edge(<sa>, <out>, "->"),
    edge(<sb>, <out>, "->"),
  ),
  caption: [Model probabilistycznej sieci neuronowej (PNN) \ Dla przejrzystości pokazano tylko 2 wzorce treningowe dla każdej z 2 klas oraz $p =2$ cechy wejściowe],
) <Diagram_PNN>

Poszczególne symbole użyte na #ref(<Diagram_PNN>, supplement: [Rysunku]) oznaczają:

- $x_1, x_2$ -- wejścia sieci (cechy danych wejściowych),
- $P_("A1"), P_("A2")$ -- neurony wzorcowe dla klasy $A$,
- $P_("B1"), P_("B2")$ -- neurony wzorcowe dla klasy $B$,
- $Sigma_A, Sigma_B$ -- neurony sumujące odpowiednio dla klas $A$ i $B$,
- Decision (Argmax) -- neuron decyzyjny przypisujący obserwację do klasy
  o najwyższej estymowanej gęstości prawdopodobieństwa.

=== Warstwa wejściowa (Input Layer)

Warstwa wejściowa nie wykonuje żadnych operacji obliczeniowych. Jej jedynym zadaniem jest
wprowadzenie wektora cech $bold(x) = [x_1, x_2, dots, x_p]^T in RR^p$ do sieci oraz
dystrybucja jego składowych do wszystkich neuronów warstwy wzorców. Liczba neuronów w tej
warstwie jest równa wymiarowości $p$ przestrzeni wejściowej $x$.

=== Warstwa wzorców (Pattern Layer) <patternlayer>

Warstwa wzorców stanowi bezpośrednią reprezentację zbioru uczącego -- każdy neuron
przechowuje dokładnie jeden wzorzec treningowy $bold(x)_(k j)$, gdzie $k$ oznacza indeks
klasy, a $j$ -- numer wzorca w obrębie tej klasy.

Dla danego wektora wejściowego $bold(x)$ neuron oblicza odległość euklidesową między
wejściem a zapamiętanym wzorcem, a następnie przetwarza ją przez funkcję jądra
#cite(<specht1990pnn>). Domyślnie stosuje się jądro Gaussa (zob. @kernel):
// TODO: Może podmienic phi na K(x, x_(k j)) żeby było bardziej intuicyjnie

$ phi_(k j)(bold(x)) = frac(1, sigma sqrt(2 pi)) exp lr(( -frac(||bold(x) - bold(x)_(k j)||^2, 2 sigma^2) )) $

gdzie:
- $||bold(x) - bold(x)_(k j)||$ -- odległość euklidesowa w przestrzeni cech (zob. _#ref(<eq_distance>)_),
- $sigma$ -- parametr wygładzania (*smoothing parameter* lub *bandwidth*),
  decydujący o szerokości funkcji dzwonowej (zob. @sigma).

Odległość euklidesową między wektorem $x = [x_1, x_2, dots, x_p]^T$ a wzorcem $bold(x)_(k j)$ można wyrazić wzorem:

$ ||bold(x) - bold(x)_(k j)|| = sqrt(sum_(i=1)^p (x_i - x_(k j i))^2) $ <eq_distance>
gdzie:
- $p$ -- liczba cech (wymiarowość przestrzeni wejściowej),
- $x_i$ -- $i$-ta cecha wektora wejściowego,
- $x_(k j i)$ -- $i$-ta cecha wzorca $bold(x)_(k j)$.
=== Warstwa sumowania (Summation Layer)

Każdy neuron sumujący agreguje wyjścia neuronów wzorcowych należących do tej samej klasy
#cite(<bishop2006prml>):

$ S_k (bold(x)) = sum_(j=1)^(N_k) phi_(k j)(bold(x)) $

//TODO: Może zmienić na $S_k (bold(x)) = frac(1, N_k) sum_(j=1)^(N_k) phi_(k j)(bold(x))$ żeby było bardziej intuicyjnie
gdzie:
- $S_k (bold(x))$ -- wynik działania neuronu sumującego dla klasy $k$,
- $N_k$ -- liczba wzorców treningowych należących do klasy $k$,
- $phi_(k j)(bold(x))$ -- wyjście neuronu wzorcowego dla wzorca $j$ w klasie $k$. (Funkcja jądra obliczona w warstwie wzorców).

=== Warstwa wyjściowa (Output Layer)

Warstwa wyjściowa implementuje optymalną regułę decyzyjną Bayesa #cite(<hastie2009elements>).
Zakładając równe prawdopodobieństwa *a priori* #footnote([Każda klasa na początku ma takie samo prawdopodobieństwo]) klas oraz jednakowe koszty błędnej
klasyfikacji, obserwacja $bold(x)$ zostaje przypisana do klasy $hat(k)$ o najwyższej
estymowanej gęstości prawdopodobieństwa

$ hat(k) = op("arg max")_k f_k (bold(x)) $

Architektura PNN zapewnia brak problemu minimów lokalnych (sieć nie jest trenowana
gradientowo) oraz asymptotyczną zbieżność do optymalnego klasyfikatora bayesowskiego
wraz ze wzrostem liczebności zbioru uczącego #cite(<specht1990pnn>).

== Funkcje jądra <kernel>

Funkcja jądra $K(bold(x), bold(x)_i)$ określa, jak silnie wzorzec $bold(x)_i$
wpływa na estymowaną gęstość w punkcie $bold(x)$ #cite(<silverman1986density>).
Wybór funkcji jądra wpływa na kształt granicy decyzyjnej i odporność modelu na szum.
Nie ma jednej uniwersalnej funkcji jądra, która byłaby najlepsza dla wszystkich problemów -- dobór powinien być dostosowany do charakterystyki danych #cite(<specht1990pnn>).

=== Gaussowska (Gaussian Kernel)

$ K(bold(x), bold(x)_i) = frac(1, sigma sqrt(2 pi)) dot exp lr(( -frac(||bold(x) - bold(x)_i||^2, 2 sigma^2) )) $

Najczęściej stosowana funkcja jądra #cite(<specht1990pnn>) #cite(<Kusy2014ProbabilisticNN>). Gładka, różniczkowalna,
symetrycznie zanika wraz z odległością. Wrażliwa na wartości odstające ze względu
na szybki zanik gaussowski.

=== Laplasjańska (Laplacian Kernel)

$ K(bold(x), bold(x)_i) = frac(1, 2 sigma) dot exp lr(( -frac(||bold(x) - bold(x)_i||, sigma) )) $

Zanika wolniej niż jądro gaussowskie -- jest mniej czułe na wartości odstające
i lepiej sprawdza się przy danych z rozkładami o grubych ogonach.

=== Cauchy'ego (Cauchy Kernel) #cite(<cursedKernels>)

$ K(bold(x), bold(x)_i) = frac(1, pi sigma) dot frac(1, 1 + frac(||bold(x) - bold(x)_i||^2, sigma^2)) $

Posiada bardzo grube ogony, przez co silnie uwzględnia odległe wzorce.
Przydatne przy danych silnie zaszumionych lub heterogenicznych.

=== Odwrotna multikwadratowa (Inverse Multiquadric Kernel) #cite(<commonKernels>)

$ K(bold(x), bold(x)_i) = frac(1, sigma) dot frac(1, sqrt(||bold(x) - bold(x)_i||^2 + sigma^2)) $

Funkcja zawsze dodatnia o łagodnym zaniku. Stosowana jako alternatywa dla jądra
gaussowskiego, gdy pożądana jest mniejsza czułość na dokładną wartość $sigma$.

=== Epanecznika (Epanechnikov Kernel) #cite(<wikipedia_kernels>)
$ K(bold(x), bold(x)_i) = frac(3, 4 sigma) dot max (lr(0, 1 - frac(||bold(x) - bold(x)_i||^2, sigma^2))) $
Jądro to jest gładkie wewnątrz obszaru wsparcia, ale na jego brzegu ma nieciągłość pochodnej. Z tego powodu dobrze sprawdza się w zadaniach, w których zależy nam na lokalnym uśrednianiu i ograniczeniu wpływu odległych obserwacji. W porównaniu z jądrem gaussowskim daje bardziej „lokalny” charakter estymacji i może lepiej tłumić słabo istotne, odległe wzorce.

Używamy funkcji $max$ do zapewnienia, że jądro nie zwróci wartości ujemnej. Ponieważ wartości ujemne nie mają sensu w kontekście estymacji gęstości, funkcja $max$ gwarantuje, że jądro będzie miało wartość zero dla obserwacji znajdujących się poza obszarem wsparcia (gdzie $||bold(x) - bold(x)_i||^2 > sigma^2$).

=== Trójkątna (Triangular Kernel)
$ K(bold(x), bold(x)_i) = frac(1, sigma) max (lr(0, 1 - frac(||bold(x) - bold(x)_i||, sigma))) $
Jądro trójkątne jest funkcją liniową, która maleje wraz z odległością, osiągając wartość zero w odległości $sigma$. Jest to jądro o ograniczonym zasięgu, które uwzględnia tylko obserwacje znajdujące się w promieniu $sigma$ od punktu estymacji. Dzięki temu jest szczególnie skuteczne w zadaniach, gdzie ważne jest uwzględnienie tylko lokalnych informacji i ograniczenie wpływu odległych obserwacji.

=== Jednostkowa (Uniform Kernel)
$ K(x, x_i) = cases(1/2 "kiedy" ||x - x_i|| <= sigma, 0 "w przeciwnym wypadku") $
Jądro jednostkowe jest funkcją o stałej wartości wewnątrz obszaru wsparcia i zerowej wartości poza nim. Jest to najprostsze jądro, które uwzględnia tylko obserwacje znajdujące się w promieniu $sigma$ od punktu estymacji, przypisując im jednakową wagę. Ze względu na swoją prostotę, jądro jednostkowe może być mniej skuteczne w zadaniach, gdzie ważne jest uwzględnienie różnic w odległości między obserwacjami a punktem estymacji.

Porównanie kształtu różnych funkcji jądra przedstawiono na #ref(<PorownanieJader>, supplement: [Rysunku]).

== Parametr wygładzania $sigma$ <sigma>

Parametr $sigma$ (parametr wygładzania, ang. *bandwidth*, *smoothing parameter*) jest kluczowym hiperparametrem
PNN -- kontroluje szerokość funkcji jądra i tym samym zasięg wpływu każdego wzorca
treningowego na estymowaną gęstość #cite(<specht1990pnn>):

- *zbyt małe $sigma$* -- sieć dopasowuje się ściśle do zbioru uczącego (overfitting),
  granica decyzyjna jest nieregularna i wrażliwa na szum,
- *zbyt duże $sigma$* -- sieć nadmiernie wygładza rozkład (underfitting),
  tracąc lokalne struktury w danych.

#figure(
  image("Images/porownaniejader2.png"),
  caption: [Porównanie kształtu funkcji jądra dla kilku wartości $sigma$],
) <PorownanieJader>


=== Dobór optymalnej wartości $sigma$ <sigmasearch>
Do wyznaczenia optymalnej wartości $sigma$ stosuje się zazwyczaj jedną z poniższych metod:

- *$k$-krotna walidacja krzyżowa* (ang. $k$-fold cross-validation) -- wartość $sigma$
  dobierana jest tak, aby minimalizować błąd klasyfikacji na zbiorze walidacyjnym
  #cite(<hastie2009elements>),
- *reguła Silvermana* #cite(<silverman1986density>) -- heurystyczny estymator oparty
  na odchyleniu standardowym danych:
  $sigma^* = 1.06 * hat(s) dot N^(-1\/5)$, gdzie $hat(s)$ <silvermanrule>
  to odchylenie standardowe
  cech, a $N$ to liczba wzorców,

- *przeszukiwanie siatki* (ang. grid search) -- systematyczne sprawdzenie zbioru
  kandydujących wartości $sigma$ z oceną na zbiorze testowym #cite(<hastie2009elements>).

W praktyce zaleca się stosowanie walidacji krzyżowej jako metody najbardziej odpornej
na specyfikę konkretnego zbioru danych.

== Różnice między PNN a innymi sieciami neuronowymi
Sieć PNN różni się od tradycyjnych sieci neuronowych, takich jak perceptron wielowarstwowy (MLP) czy sieci konwolucyjne (CNN), przede wszystkim sposobem reprezentacji i przetwarzania danych:
- *Brak klasycznego uczenia gradientowego* -- PNN nie wymaga iteracyjnego dostosowywania wag, dzięki czemu nie występuje problem minimów lokalnych, a budowa modelu przebiega szybciej,
- *Bezpośrednia reprezentacja zbioru uczącego* -- każdy neuron wzorcowy odpowiada dokładnie jednemu przykładowi treningowemu, co pozwala od razu wykorzystać cały zbiór danych bez konieczności jego kompresji,
- *Estymacja gęstości zamiast funkcji decyzyjnej* -- PNN opiera się na estymacji gęstości prawdopodobieństwa klas, natomiast MLP i CNN uczą się bezpośrednio funkcji decyzyjnej, co wpływa na kształt granic decyzyjnych oraz odporność modelu na szum.

PNN jest szczególnie skuteczna w zadaniach klasyfikacji, w których dostępna jest niewielka liczba danych, ponieważ tradycyjne sieci mogą mieć wtedy trudności z generalizacją. Z drugiej strony, ze względu na bezpośrednią reprezentację zbioru uczącego, PNN może być niepraktyczna w przypadku bardzo dużych zbiorów danych, głównie z uwagi na wysokie wymagania pamięciowe oraz dłuższy czas predykcji.

= Analiza danych
Zbiór danych spambase #cite(<spambase>) zawiera 4601 próbek wiadomości e-mail, z których 57% to spam. Każda próbka jest reprezentowana przez 57 cech numerycznych, które opisują różne aspekty wiadomości, takie jak częstotliwość występowania określonych słów i znaków specjalnych, długość wiadomości. Celem jest klasyfikacja wiadomości jako spam lub nie-spam na podstawie tych cech.

#figure(
  table(
    columns: (auto, auto, 1fr, auto),
    stroke: 0.5pt,
    fill: (col, row) => if row == 0 { rgb("#2c3e50") } else if calc.odd(row) { rgb("#f2f2f2") } else { white },
    inset: 7pt,
    align: (col, row) => if col == 0 or col == 3 { center } else { left },
    // Nagłówek
    table.cell(fill: rgb("#2c3e50"))[#text(fill: white, weight: "bold")[Nr]],
    table.cell(fill: rgb("#2c3e50"))[#text(fill: white, weight: "bold")[Nazwa atrybutu]],
    table.cell(fill: rgb("#2c3e50"))[#text(fill: white, weight: "bold")[Opis]],
    table.cell(fill: rgb("#2c3e50"))[#text(fill: white, weight: "bold")[Zakres wartości]],

    // word_freq_*
    [1--48],
    [word\_freq\_SŁOWO],
    [Procentowy udział danego słowa (make, free, you, credit, money, hp itp.) w całkowitej liczbie słów wiadomości. Obliczane jako $100 dot n_"słowo" / n_"słów"$. Zbiór zawiera 48 różnych słów charakterystycznych dla spamu i poczty służbowej.],
    [$[0, 100]$],

    // char_freq_*
    [49--54],
    [char\_freq\_ZNAK],
    [Procentowy udział danego znaku (; ( \[ ! \$ \#) w całkowitej liczbie znaków wiadomości. Obliczane jako $100 dot n_"znak" / n_"znaków"$. Znaki takie jak ! i \$ są silnymi wskaźnikami spamu.],
    [$[0, 100]$],

    // capital_run_length_*
    [55], [capital\_run\_length\_average],
    [Średnia długość nieprzerwanych sekwencji wielkich liter (wartość rzeczywista)],
    [$[1, 1102.5]$],
    [56], [capital\_run\_length\_longest],
    [Długość najdłuższej nieprzerwanej sekwencji wielkich liter (wartość całkowita)],
    [$[1, 9989]$],
    [57], [capital\_run\_length\_total],
    [Łączna liczba wielkich liter w całej wiadomości (wartość całkowita)],
    [$[1, 15841]$],

    // Zmienna docelowa
    [58], [spam],
    [Klasa wiadomości: *1* = wiadomość jest spamem, *0* = wiadomość nie jest spamem],
    [$\{0,1\}$],
  ),
  caption: [Opis cech zbioru danych spambase @spambase],
)
Zbiór nie zawiera brakujących wartości, a cechy są numeryczne, co ułatwia ich bezpośrednie wykorzystanie w modelu PNN.

#figure(
  ```py
  def load_data(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load data from a CSV file. The last column is assumed to be the class label.
    :param path: Path to the CSV file containing the data.
    :return: A tuple containing the features (X) and labels (y) as numpy arrays.
    """
    data = np.loadtxt(path, delimiter=",")
    x = data[:, :-1]  # All columns except the last one are features
    y = data[:, -1]  # The last column is the class label
    return x, y
  ```,
  caption: [Metoda odpowiedzialna za wczytywanie danych z pliku `.data`],
)
Dane pobierane są z pliku `.data`. Wszystkie cechy poza ostatnią są zwracane jako macierz z wymiarami $N times p$, gdzie $N$ to liczba próbek, a $p$ to liczba cech (57). Ostatnia kolumna zawiera etykiety klas (0 lub 1) i jest zwracana jako wektor o długości $N$.

Dane przed przetwarzaniem są normalizowane, ponieważ cechy zbioru danych mają różne zakresy wartości (częstość występowania słów to zakres od $0$ do $100$, natomiast  długość wiadomości to zakres od $0$ do $10000$).
Jeżeli nie przeprowadzimy normalizacji, odległości euklidesowe obliczane przez PNN będą zdominowane przez cechy o większych zakresach, co może prowadzić do błędnych klasyfikacji. Normalizacja zapewnia, że wszystkie cechy mają równy wpływ na estymację gęstości i poprawia wydajność modelu.

#figure(
  ```py
  def normalize_data(X: np.ndarray) -> np.ndarray:
    """
    Normalize the features in the dataset to have zero mean and unit variance.
    :param X: The input feature matrix to be normalized.
    :return: The normalized feature matrix.
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std
  ```,
  caption: [Część skryptu odpowiedzialna za normalizację danych],
)
Po znormalizowaniu danych, każda cecha ma wartość średnią równą $0$ i odchylenie standardowe równe $1$. Dzięki temu PNN może efektywnie obliczać odległości euklidesowe i estymować gęstość prawdopodobieństwa bez dominacji jednej cechy nad innymi.

= Skrypt programu
#figure(
  ```py
    class PNN:
    """
    Probabilistic Neural Network (PNN) implementation for classification tasks.
    Train with fit() method and predict with predict() method. Uses a kernel function to compute similarity between input and training patterns.
    """

    def __init__(self, kernel: Callable[[np.ndarray, np.ndarray, float], np.ndarray] = PnnKernels.gaussian_kernel, sigma=0.1) -> None:
        """
        Initialize the PNN model with an optional kernel function and sigma parameter.
        :param kernel: The kernel function to use to compute similarity between input and training patterns. Default is the Gaussian kernel.
        :param sigma: The sigma parameter for the kernel function (default is 0.1).
        """
        self.patterns : Optional[dict] = None
        self.sigma: float = sigma
        self.kernel = kernel
  ```,
  caption: [Klasa implementująca probabilistyczną sieć neuronową (PNN)],
)
Parametr ```py self.kernel``` odpowiada funkcji jądra używanej do estymacji gęstości, musi to być funkcja która przyjmuje trzy argumenty: wektor wejściowy, wektor wzorca oraz wartość parametru wygładzania $sigma$. Domyślnie jest to jądro Gaussa.

```py self.sigma``` to parametr wygładzania $sigma$ kontrolujący szerokość funkcji jądra z domyślną wartością $0.1$.

Słownik ```py self.patterns``` będzie przechowywał wzorce treningowe pogrupowane według klas.

#figure(
  ```py
  def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Fit (Train) the PNN model to the training data by storing the patterns for each class.
        :param x: Training data features
        :param y: Training data labels corresponding to the features in x.
        :return: None
        """
        self.patterns = {c: x[y == c] for c in np.unique(y)}
  ```,
  caption: [Metoda `fit()` odpowiedzialna za uczenie się sieci poprzez przypisanie wzorców do klas],
)
Metoda ```py def fit()``` jest odpowiedzialna za uczenie się sieci. Przypisuje ona każdej klasie jej wzorce, tworząc słownik, w którym kluczem jest etykieta klasy, a wartością jest macierz zawierająca wzorce treningowe należące do tej klasy.

#figure(
  ```py
  def predict_single(self, x: np.ndarray) -> Optional[float]:
        """
        Predict the class label for a single input
        :param x: Input vector
        :return: Predicted class label
        """
        if self.patterns is None:
            raise ValueError("Model has not been fitted yet.")
        best_class : Optional[float] = None
        best_score = -np.inf
        for cls, patterns in self.patterns.items():
            score = np.sum(self.kernel(x, patterns, self.sigma))
            if score > best_score:
                best_score = score
                best_class = cls
        return best_class
  ```,
  caption: [Metoda `predict_single()` odpowiedzialna za przewidywanie klasy dla pojedynczej obserwacji],
)

Metoda ```py def predict_single()``` oblicza estymowaną gęstość prawdopodobieństwa dla każdej klasy, sumując wartości funkcji jądra między wejściem a wzorcami danej klasy. Klasa z najwyższym wynikiem jest zwracana jako przewidywana etykieta.

#figure(
  ```py
      def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict the class labels for a set of input vectors.
        :param x: Array of input vectors to predict class labels for.
        :return: Array of predicted class labels corresponding to the input vectors.
        """
        return np.array([self.predict_single(xi) for xi in x])
  ```,
  caption: [Pomocnicza metoda `predict()`],
)
Metoda ```py def predict()``` jest pomocniczą funkcją, która umożliwia przewidywanie klas dla wielu obserwacji jednocześnie. Dla każdego wektora wejściowego w macierzy $x$ wywołuje metodę `predict_single` i zwraca tablicę z przewidywanymi etykietami klas.

#figure(
  block(
    align(left, raw(read("../src/kernels.py"), lang: "python")),
    breakable: true,
    width: 100%,
  ),
  caption: [Klasa implementująca funkcje jądra używane w probabilistycznej sieci neuronowej z _#ref(<kernel>, supplement: [Rozdziału])_],
  kind: raw,
  placement: none,
)
Każda metoda z ```py class PNNKernels``` implementuje inną funkcję jądra, która jest używana do obliczania podobieństwa między wektorem wejściowym a wzorcami treningowymi.
Każda funkcja jądra jest zaimplementowana w osobnej metodzie klasy ```py class PnnKernels```, co pozwala na łatwe dodawanie nowych funkcji jądra i eksperymentowanie z różnymi konfiguracjami. W każdej z tych funkcji korzystamy z `numpy` #cite(<numpydocs>) aby przyśpieszyć obliczenia i umożliwić efektywne przetwarzanie dużych zbiorów danych.

Pełny kod projektu dostępny jest w _#link(<Dodatki>, [dodatkach])_
= Eksperymenty

== Badanie wpływu funkcji jądra oraz parametru wygładzania $sigma$
W Probablistycznej sieci neuronowej kluczową rolę odgrywa parametr $sigma$ (_#ref(<sigma>)_), który kontroluje szerokość funkcji jądra używanej do estymacji gęstości. Różne wartości $sigma$ mogą znacząco wpłynąć na dokładność klasyfikacji, dlatego ważne jest przeprowadzenie eksperymentów w celu znalezienia optymalnej wartości tego parametru.
Porównamy również różne funkcje jądra, aby zobaczyć, jak wpływają one na wydajność modelu. (_#ref(<kernel>)_).

=== Dobór optymalnej wartości $sigma$ i funkcji jądra za pomocą walidacji krzyżowej
Aby znaleźć najlepszą konfigurację jądra i wartości $sigma$, przeprowadzimy eksperyment z walidacją krzyżową. Będziemy testować wartości $sigma$ w zakresie od $0.001$ do $2$ z krokiem $0.005$, co pozwoli nam zobaczyć, jak dokładność zmienia się w szerokim zakresie wartości tego parametru.
Dla każdej kombinacji funkcji jądra i wartości $sigma$ obliczymy dokładność klasyfikacji na zbiorze walidacyjnym i porównamy wyniki, aby znaleźć optymalną konfigurację.

#figure(
  ```py
    def kFold_search(
      x: np.ndarray,
      y: np.ndarray,
      min_sigma: float = 0.001,
      max_sigma: float = 2,
      diff_sigma: float = 0.005,
      splits: int = 5,
      kerns=[PnnKernels.gaussian_kernel],
  ) -> dict[str, dict[float, np.ndarray]]:
      """
      Perform k-fold cross-validation to search for the best sigma and kernel configuration.
      :param x: The input feature matrix.
      :param y: The target labels.
      :param min_sigma: The minimum value for sigma.
      :param max_sigma: The maximum value for sigma.
      :param diff_sigma: The step size for sigma.
      :return: A dictionary containing the accuracies for each kernel and sigma combination.
      """
      accuracies: dict[
          str, dict[float, np.ndarray]
      ] = {}  # {kernel_name: {sigma_value: (min_acc, max_acc, avg_acc)}}

      kf = KFold(n_splits=splits, shuffle=True, random_state=67)
      p = PNN()
      for kernel in kerns:
          kernel_name = kernel.__name__
          accuracies[kernel_name] = {}
          for sigma in np.arange(min_sigma, max_sigma + diff_sigma, diff_sigma):
              p.kernel = kernel
              p.sigma = sigma
              fold_accuracies = []
              for train_index, test_index in kf.split(x):
                  x_train, x_test = x[train_index], x[test_index]
                  y_train, y_test = y[train_index], y[test_index]
                  p.fit(x_train, y_train)
                  y_pred = p.predict(x_test)
                  fold_accuracy = np.mean(y_pred == y_test)
                  fold_accuracies.append(fold_accuracy)
              accuracies[kernel_name][sigma] = np.array(fold_accuracies)
              print(
                  f"Kernel: {kernel_name}, Sigma: {sigma:.3f}, Accuracy: {np.mean(accuracies[kernel_name][sigma]):.4f} (min: {np.min(accuracies[kernel_name][sigma]):.4f}, max: {np.max(accuracies[kernel_name][sigma]):.4f})"
              )
      return accuracies
  ```,
  caption: [Część skryptu odpowiedzialna za poszukiwanie najlepszej konfiguracji jądra i $sigma$ z użyciem walidacji krzyżowej (zob. _#ref(<sigmasearch>)_)],
)
K-krotna walidacja jest zaimplementowana za pomocą klasy `KFold` z biblioteki `sklearn.model_selection` #cite(<skFold>) która dzieli dane na n losowych podzbiorów, w naszym przypadku $n=5$. Dla każdej kombinacji funkcji jądra i wartości $sigma$ model jest trenowany na $n-1$ podzbiorach, a następnie testowany na pozostałym podzbiorze. Proces ten jest powtarzany dla wszystkich możliwych podziałów danych, a dokładność jest obliczana jako średnia z wyników uzyskanych na wszystkich podziałach.

Badamy wartości $sigma$ w zakresie od $0.001$ do $2$ z krokiem $0.005$, co pozwala nam zobaczyć, jak dokładność zmienia się w szerokim zakresie wartości tego parametru.

#figure(
  image("Images/pnn2.png", width: 95%),
  caption: [Wykres dokładności dla różnych konfiguracji jądra i wartości sigma. \ Każda linia reprezentuje inną funkcję jądra, a oś x przedstawia wartości sigma. Każdy punkt na wykresie to średnia dokładność uzyskana z walidacji krzyżowej dla danej kombinacji jądra i sigma.],
)
Najlepszą dokładność klasyfikacji osiągnięto dla *jądra laplasjańskiego przy $sigma = 0.611$*, przy średniej dokładności *$0.9244$* (min: $0.9121$, max: $0.9359$). Ze względu na podobną charakterystykę jądra gaussowskiego, osiągnęło ono podobne wyniki, największa średnia dokładność dla jądra gaussowskiego wyniosła $0.9146$ (min: $0.9088$, max: $0.9250$).

Jądra trójkątne, epańczykowa i jednostkowe wykazały prawie taką samą dokładność na całym zakresie $sigma$, co może być spowodowane ich podobną charakterystyką (zob. _#ref(<PorownanieJader>)_). Najlepszy wynik z tych trzech funkcji uzyskało jądro trójkątne, które osiągnęło dokładność $0.8213$ przy $sigma approx 2$. Co jest znacznie gorszym wynikiem niż jądra gaussowskie i laplasjańskie.

Jądra cauchy'ego i odwrotna multikwadratowa nie są odpowiednie dla tego zbioru danych, ich dokładność nie przekroczyła $0.8$ dla żadnej wartości $sigma$.

=== Wykorzystanie reguły Silvermana do oszacowania optymalnej wartości $sigma$
Wykorzystaliśmy również regułę Silvermana (_#ref(<sigmasearch>)_) do oszacowania optymalnej wartości $sigma$ na podstawie odchylenia standardowego cech.

#figure(
  ```py
      silverman = 1.06 * np.std(x) * (len(x) ** (-1 / 5))
  ```,
  caption: [Część skryptu odpowiedzialna za obliczenie wartości $sigma$ na podstawie reguły Silvermana],
)

Obliczona wartość $sigma^* = 0.1962$

/*
Silverman's rule of thumb sigma: 0.1962
Silverman's rule of thumb accuracies:
Kernel: gaussian_kernel, Accuracy: 0.9033
Kernel: laplacian_kernel, Accuracy: 0.9141
Kernel: cauchy_kernel, Accuracy: 0.7511
Kernel: inverse_multiquadric_kernel, Accuracy: 0.6043
Kernel: epanechnikov_kernel, Accuracy: 0.7413
Kernel: triangular_kernel, Accuracy: 0.7413
Kernel: uniform_kernel, Accuracy: 0.7413
*/

#figure(
  table(
    columns: (auto, auto),
    stroke: 0.5pt,
    fill: (col, row) => if row == 0 { rgb("#2c3e50") } else if calc.odd(row) { rgb("#f2f2f2") } else { white },
    inset: 7pt,
    align: (col, row) => center,
    // Nagłówek
    table.cell(fill: rgb("#2c3e50"))[#text(fill: white, weight: "bold")[Jądro]],
    table.cell(fill: rgb("#2c3e50"))[#text(fill: white, weight: "bold")[Dokładność]],

    [Gaussowska], [$0.9033$],
    [Laplasjańska], [$0.9141$],
    [Cauchy'ego], [$0.7511$],
    [Odwrotna multikwadratowa], [$0.6043$],
    [Epanecznika], [$0.7413$],
    [Trójkątna], [$0.7413$],
    [Jednostkowa], [$0.7413$],
  ),
  caption: [Dokładność klasyfikacji dla różnych funkcji jądra przy wartości $sigma$ oszacowanej za pomocą reguły Silvermana],
)

Ponownie najlepszą dokładność osiągnęło jądro laplasjańskie. Dokładność jest mniejsza niż w przypadku optymalnego $sigma$ znalezionego za pomocą walidacji krzyżowej, ale nadal jest wysoka ($0.9141$). \
$Delta = 0.9244 - 0.9141 = 0.0103$, więc różnica jest niewielka, co potwierdza, że reguła Silvermana jest użytecznym narzędziem do oszacowania początkowej wartości $sigma$, która następnie może być dostrojona za pomocą walidacji krzyżowej.

== Badanie wpływu stałej normalizującej w funkcjach jądra
W prawie każdej definicji funkcji jądra uwzględnione są stałe normalizujące, które zapewniają, że jądro jest poprawnie znormalizowane jako estymator gęstości.
Na przykład dla jądra gaussowskiego, stała normalizująca to $frac(1, sigma sqrt(2 pi))$.
Jednak w kontekście klasyfikacji, gdzie ostateczna decyzja opiera się na maksymalnej wartości estymowanej gęstości, te stałe nie powinny wpływać na wynik klasyfikacji, ponieważ są one takie same dla wszystkich klas i nie zmieniają relatywnych wartości estymowanych gęstości.
Możemy przeprowadzić eksperyment, w którym porównamy dokładność klasyfikacji z uwzględnieniem stałych normalizujących i bez nich. Oczekujemy, że dokładność pozostanie taka sama, co potwierdzi, że te stałe nie mają wpływu na decyzję klasyfikacyjną.

Dane sprawdzimy dla jądra gaussowskiego i parametru $sigma in {0.001, 0.101, 0.201, dots, 2}$ \
Nieznormalizowane jądro gaussowskie będzie miało postać:
$ K(bold(x), bold(x)_i) = exp(-frac(||bold(x) - bold(x)_i||^2, 2 sigma^2)) $
#figure(
  ```py
  # Check if you need normalization in kernels
    kernels = [
        PnnKernels.gaussian_kernel,
        lambda x, y, sigma: np.exp(-np.sum((x - y) ** 2, axis=1) / (2 * sigma ** 2)),  # Gaussian without normalization
    ]

    accs = kFold_search(x, y, kerns=kernels, max_sigma=1, diff_sigma=0.1)
    gauss_acc = accs[kernels[0].__name__]
    no_norm_acc = accs[kernels[1].__name__]
    # First ensure the same set of sigma keys, then compare arrays per key using allclose.
    same_keys = set(gauss_acc.keys()) == set(no_norm_acc.keys())
    if not same_keys:
        is_same = False
    else:
        is_same = all(
            np.allclose(gauss_acc[sigma], no_norm_acc[sigma])
            for sigma in gauss_acc.keys()
        )
    print(f"Gaussian kernel (with normalization) equals no-norm Gaussian? {is_same}")
  ```,
  caption: [Część skryptu odpowiedzialna za porównanie dokładności klasyfikacji z uwzględnieniem stałych normalizujących funkcje jądra i bez nich],
)

Ponownie używamy metody `kFold_search` do porównania dokładności klasyfikacji dla jądra gaussowskiego z normalizacją i bez niej. Sprawdzamy, czy dokładności są takie same dla wszystkich wartości $sigma$. Robimy to poprzez porównanie tablic dokładności dla obu wariantów jądra i sprawdzenie, czy są one bliskie (używając `np.allclose`). Jeśli dokładności są takie same dla wszystkich wartości $sigma$, to potwierdza, że stałe normalizujące nie mają wpływu na dokładność klasyfikacji.


Po uruchomieniu skryptu otrzymujemy:
#figure(
  ```
    Gaussian kernel (with normalization) equals no-norm Gaussian? True
  ```,
)
Co potwierdza, że stałe normalizujące nie mają wpływu na dokładność klasyfikacji, ponieważ oba warianty jądra dają taką samą dokładność.



= Podsumowanie i wnioski
//TODO: Też napisać lepiej niż LLM

W ramach projektu przedstawiono teoretyczne podstawy probabilistycznej sieci neuronowej (PNN), jej architekturę oraz rolę funkcji jądra i parametru $sigma$. Następnie przeprowadzono eksperyment porównujący kilka funkcji jądra dla różnych wartości $sigma$.

Uzyskane wyniki potwierdzają, że skuteczność PNN silnie zależy od doboru hiperparametrów. Najlepszą dokładność klasyfikacji osiągnięto dla jądra laplasjańskiego przy odpowiednio dobranym $sigma$, natomiast pozostałe jądra wykazywały większą wrażliwość lub niższą stabilność wyników.

Reguła Silvermana okazała się użytecznym narzędziem do oszacowania początkowej wartości $sigma$, która następnie mogła być dostrojona za pomocą walidacji krzyżowej, co potwierdziło jej praktyczną wartość w kontekście PNN.

Sprawdziliśmy również, że stałe normalizujące w funkcjach jądra nie wpływają na dokładność klasyfikacji, co jest zgodne z teoretycznymi oczekiwaniami, ponieważ decyzja klasyfikacyjna opiera się na maksymalnej wartości estymowanej gęstości, a te stałe są takie same dla wszystkich klas.

Probabilistyczne sieci neuronowe są skuteczną metodą klasyfikacji, podczas naszych eksperymentów osiągnęły wysoką dokładność porównywalną z innymi metodami klasyfikacji #cite(<spambase>). Mimo bardzo krótkiego czasu uczenia, PNN może być konkurencyjną alternatywą dla bardziej złożonych modeli, zwłaszcza w zadaniach z niewielką ilością danych.

Najważniejsze wnioski:
- dobór funkcji jądra ma istotny wpływ na jakość klasyfikacji,
- parametr $sigma$ powinien być strojony eksperymentalnie (np. walidacją krzyżową),
- PNN jest prostą i skuteczną metodą dla zadań klasyfikacji, szczególnie przy mniejszych zbiorach danych,
- ograniczeniem PNN pozostaje koszt pamięciowy i obliczeniowy przy dużej liczbie wzorców.

#bibliography("bibliography.bib")

#appendix([
  = Dodatki <Dodatki>
  #enum(
    [Repozytorium z kodem źródłowym projektu dostępne na #link("https://github.com/NimVrod/PNN_S")],
    numbering: "[1]",
  )
])
