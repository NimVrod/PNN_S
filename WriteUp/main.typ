#set page(paper: "a4", margin: 2cm,
footer: context [
    #grid(
      columns: (1fr, 1fr, 1fr),    // Creates 3 equal zones
      align: (left, center, right), // Aligns content within those zones

      [], // 1. EMPTY LEFT (Crucial for balancing)

      // 2. CENTER Page Number
      smallcaps[#context counter(page).display()],

      // 3. RIGHT Location & Date
      smallcaps[Rzeszów #datetime.today().display()]
    )
  ],
)

#set heading(numbering: "1.1.a.")
#set text(lang: "pl")



//Title page
#page(numbering: none, header: none, footer: none)[
  //TODO: Add prz images

  #align(center)[
    #text(weight: "extrabold", size: 1cm)[Sztuczna Inteligencja] //Subject

    #text(size: 0.5cm)[Projekt] // Laboratory number

  #text(size: 0.5cm)[Probabilistyczna sieć neuronowa] // Laboratory subject

  #align(bottom)[
      Hubert Dec $-$ 179474 $-$ L04 //Author information
      #linebreak()
      #datetime.today().display()
  ]

  ]
]

#outline()
#pagebreak()

= Opis problemu

= Część teoretyczna
Probabilistyczna sieć neuronowa (PNN) #cite(<specht1990pnn>) jest rodzajem sieci neuronowej, która wykorzystuje funkcje jądra do modelowania rozkładu prawdopodobieństwa danych. PNN jest oparta na teorii Bayesa i jest często stosowana do klasyfikacji i regresji. Sieć składa się z trzech warstw: warstwy wejściowej, warstwy ukrytej i warstwy wyjściowej. Warstwa ukryta wykorzystuje funkcje jądra do obliczania podobieństwa między danymi wejściowymi a wzorcami treningowymi, co pozwala na klasyfikację danych. PNN jest szybka w uczeniu się i może być skuteczna w przypadku dużych zbiorów danych, ale może być podatna na nadmierne dopasowanie, jeśli nie jest odpowiednio regularyzowana.
PNN jest szybsza w uczeniu się niż tradycyjne sieci neuronowe, ponieważ nie wymaga iteracyjnego procesu uczenia. Zamiast tego, PNN wykorzystuje funkcje jądra do obliczania podobieństwa między danymi wejściowymi a wzorcami treningowymi, co pozwala na szybką klasyfikację danych. Jednakże, PNN może być podatna na nadmierne dopasowanie, jeśli nie jest odpowiednio regularyzowana, co może prowadzić do słabej generalizacji na nowych danych.


== Model neuronu
== Opis matematyczny sztucznego neuronu
== Sieci neuronowe jednokierunkowe wielowarstwowe
== Algorytm wstecznej propagacji błędu z przyśpieszeniem metodą adaptacyjnego współczynnika uczenia
== Metoda momentum

= Analiza danych

= Skrypt programu

= Eksperymenty
== Wyznaczenie optymalnych wartości K1 oraz K2
== Wyznaczenie optymalnych wartości lr_inc i lr_dec
== Eksperyment dla najlepszych wartości er oraz mc

= Podsumowanie i wnioski

#bibliography("bibliography.bib")
