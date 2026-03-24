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
#cite(<specht1990pnn>)

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