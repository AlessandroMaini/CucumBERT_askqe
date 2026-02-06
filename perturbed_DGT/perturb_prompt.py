perturb_synonym_fr = """Task: You will be given a {{target_lang}} sentence and your goal is to perturb the sentence by replacing one or two words (noun, verb, adjective or adverb) to its synonym. Please make sure it does not changes the meaning in a significant way. Just output the perturbed {{target_lang}} sentence without giving any additional explanation.

*** Example Starts ***
Original: Le présent contrat peut être résilié par l'une ou l'autre des parties moyennant un préavis écrit.
Perturbed: Le présent accord peut être résilié par l'une ou l'autre des parties moyennant un préavis écrit.

Original: Le Prestataire s'engage à garantir la confidentialité des informations transmises par le Client.
Perturbed: Le Prestataire s'oblige à assurer la confidentialité des données transmises par le Client.
*** Example Ends ***

Original: {{original}}
Perturbed: """


perturb_spelling_fr = """Taks: You will be given a {{target_lang}} sentence and your goal is to perturb the sentence by making spelling of one or two words wrong. The words should be important words in the sentence but not words like "le", "et", "la" or "des". Just output the perturbed {{target_lang}} sentence without giving any additional explanation.

*** Example Starts ***
Original: Le tribunal compétent sera celui du siège social de la société prestataire.
Perturbed: Le triunal compétent sera celui du siège social de la soiété prestataire.

Original: Les présentes conditions générales de vente s'appliquent à toute commande passée.
Perturbed: Les présentes conditios générales de vente s'appliquet à toute commande passée.
*** Example Ends ***

Original: {{original}}
Perturbed: """


perturb_expansion_noimpact_fr = """Task: You will be given a {{target_lang}} sentence and your goal is to perturb the sentence by adding one or two words in the sentence. Do not add words that change the intensity of the existing word. Please make sure that the added word does not disturb the grammaticality of the sentence and does not changes the meaning in a significant way. The added words would add more context that was already obvious from the sentence. Just output the perturbed {{target_lang}} sentence without giving any additional explanation.

*** Example Starts ***
Original: Tout litige relatif à l'exécution du présent contrat sera soumis au tribunal compétent.
Perturbed: Tout litige juridique relatif à l'exécution du présent contrat écrit sera soumis au tribunal compétent.

Original: Les données collectées ne seront pas transmises à des tiers sans votre consentement.
Perturbed: Les données personnelles collectées ne seront pas transmises à des tiers externes sans votre consentement.
*** Example Ends ***

Original: {{original}}
Perturbed: """


perturb_expansion_impact_fr = """Task: You will be given a {{target_lang}} sentence and your goal is to perturb the sentence by adding words in the sentence. Please make sure that the added word does not disturb the grammaticality of the sentence but should change the meaning in a significant way. Just output the perturbed {{target_lang}} sentence without giving any additional explanation.

*** Example Starts ***
Original: Le Prestataire est tenu pour responsable des dommages directs subis par le Client.
Perturbed: Le Prestataire est tenu pour responsable des dommages directs, indirects e immatériels subis par le Client.

Original: Les informations confidentielles désignent les secrets de fabrication e les données clients.
Perturbed: Les informations confidentielles désignent les secrets de fabrication, les stratégies commerciales e les données clients.
*** Example Ends ***

Original: {{original}}
Perturbed: """


perturb_alteration_fr = """Task: You will be given a {{target_lang}} sentence and your goal is to perturb the sentence by changing the affirmative sentence into negation (vice versa) or changing one word (noun, verb, adjective or adverb) to its antonym or completely different word. Please make sure that the perturbation does not disturb the grammaticality of the sentence but should change the meaning in a significant way. Just output the perturbed {{target_lang}} sentence without giving any additional explanation.

*** Example Starts ***
Original: Cette clause de non-concurrence est considérée comme valide et exécutoire par les tribunaux compétents.
Perturbed: Cette clause de non-concurrence est considérée comme nulle et exécutoire par les tribunaux compétents.

Original: Le locataire a payé l'intégralité du dépôt de garantie au moment de la signature du bail.
Perturbed: Le locataire n'a pas payé l'intégralité du dépôt de garantie au moment de la signature du bail.

Original: Souhaitez-vous confirmer l'exactitude de ces déclarations ?
Perturbed: Souhaitez-vous ignorer l'exactitude de ces déclarations ?
*** Example Ends ***

Original: {{original}}
Perturbed: """


perturb_synonym_es = """Task: You will be given a {{target_lang}} sentence and your goal is to perturb the sentence by replacing one or two words (noun, verb, adjective or adverb) to its synonym. Please make sure it does not changes the meaning in a significant way. Just output the perturbed {{target_lang}} sentence without giving any additional explanation.

*** Example Starts ***
Original: El presente contrato podrá ser rescindido por cualquiera de las partes mediante notificación previa.
Perturbed: El presente acuerdo podrá ser finalizado por cualquiera de las partes mediante notificación previa.

Original: El demandante debe presentar las pruebas pertinentes ante el tribunal competente.
Perturbed: El actor debe presentar las evidencias pertinentes ante el tribunal competente.
*** Example Ends ***

Original: {{original}}
Perturbed: """


perturb_spelling_es = """Taks: You will be given a {{target_lang}} sentence and your goal is to perturb the sentence by making spelling of one or two words wrong. The words should be important words in the sentence but not words like "le", "et", "la" or "des". Just output the perturbed {{target_lang}} sentence without giving any additional explanation.

*** Example Starts ***
Original: El juez dictó una sentencia que favorece a la parte demandante.
Perturbed: El juez dictó una setencia que favorece a la parte demadante.

Original: Este artículo regula la protección de los datos de carácter personal.
Perturbed: Este artíulo regula la protección de los datos de carácer personal.
*** Example Ends ***

Original: {{original}}
Perturbed: """


perturb_expansion_noimpact_es = """Task: You will be given a {{target_lang}} sentence and your goal is to perturb the sentence by adding one or two words in the sentence. Do not add words that change the intensity of the existing word. Please make sure that the added word does not disturb the grammaticality of the sentence and does not changes the meaning in a significant way. The added words would add more context that was already obvious from the sentence. Just output the perturbed {{target_lang}} sentence without giving any additional explanation.

*** Example Starts ***
Original: Cualquier controversia derivada de este contrato será resuelta por los tribunales competentes.
Perturbed: Cualquier controversia jurídica derivada de este contrato escrito será resuelta por los tribunales competentes.

Original: Los datos recopilados no se compartirán con terceros sin su consentimiento expreso.
Perturbed: Los datos personales recopilados no se compartirán con terceros ajenos sin su consentimiento expreso.
*** Example Ends ***

Original: {{original}}
Perturbed: """


perturb_expansion_impact_es = """Task: You will be given a {{target_lang}} sentence and your goal is to perturb the sentence by adding words in the sentence. Please make sure that the added word does not disturb the grammaticality of the sentence but should change the meaning in a significant way. Just output the perturbed {{target_lang}} sentence without giving any additional explanation.

*** Example Starts ***
Original: El Proveedor será responsable de los daños directos causados por el incumplimiento del contrato.
Perturbed: El Proveedor será responsable de los daños directos, indirectos y del lucro cesante causados por el incumplimiento del contrato.

Original: La licencia otorgada permite el uso del software en un único dispositivo del Usuario.
Perturbed: La licencia otorgada permite el uso del software en un único dispositivo del Usuario y la redistribución pública de su código fuente.
*** Example Ends ***

Original: {{original}}
Perturbed: """


perturb_alteration_es = """Task: You will be given a {{target_lang}} sentence and your goal is to perturb the sentence by changing the affirmative sentence into negation (vice versa) or changing one word (noun, verb, adjective or adverb) to its antonym or completely different word. Please make sure that the perturbation does not disturb the grammaticality of the sentence but should change the meaning in a significant way. Just output the perturbed {{target_lang}} sentence without giving any additional explanation.

*** Example Starts ***
Original: Las partes acuerdan que los términos de esta negociación serán de carácter público.
Perturbed: Las partes acuerdan que los términos de esta negociación serán de carácter confidencial.

Original: El juez ha admitido las pruebas documentales aportadas por la parte demandada.
Perturbed: El juez no ha admitido las pruebas documentales aportadas por la parte demandada.

Original: ¿Desea usted ratificar su firma en este documento legal?
Perturbed: ¿Desea usted pintar su firma en este documento legal?
*** Example Ends ***

Original: {{original}}
Perturbed: """


prompts = {
    # French
    "synonym_fr": perturb_synonym_fr,
    "spelling_fr": perturb_spelling_fr,
    "expansion_noimpact_fr": perturb_expansion_noimpact_fr,
    "expansion_impact_fr": perturb_expansion_impact_fr,
    "alteration_fr": perturb_alteration_fr,

    # Spanish
    "synonym_es": perturb_synonym_es,
    "spelling_es": perturb_spelling_es,
    "expansion_noimpact_es": perturb_expansion_noimpact_es,
    "expansion_impact_es": perturb_expansion_impact_es,
    "alteration_es": perturb_alteration_es,
}