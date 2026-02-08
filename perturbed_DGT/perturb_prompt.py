perturb_synonym_fr = """Task: You will be given a {{target_lang}} sentence and your goal is to perturb the sentence by replacing one or two words (noun, verb, adjective or adverb) to its synonym. Please make sure it does not changes the meaning in a significant way. Just output the perturbed {{target_lang}} sentence without giving any additional explanation.

*** Example Starts ***
Original: Le présent contrat peut être résilié par l'une ou l'autre des parties moyennant un préavis écrit.
Perturbed: Le présent accord peut être résilié par l'une ou l'autre des parties moyennant un préavis écrit.

Original: Le Prestataire s'engage à garantir la confidentialité des informations transmises par le Client.
Perturbed: Le Prestataire s'oblige à assurer la confidentialité des données transmises par le Client.
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


perturb_omission_fr = """Task: You will be given a {{target_lang}} sentence and your goal is to perturb the sentence by removing information from the sentence. Remove only one or two words from the sentence. Please make sure that the removed information does not disturb the grammaticality of the sentence but should change the meaning in a significant way. Just output the perturbed {{target_lang}} sentence without giving any additional explanation.

*** Example Starts ***
Original: Le licencié est autorisé à reproduire et distribuer le logiciel original.
Perturbed: Le licencié est autorisé à reproduire le logiciel original.

Original: Cette garantie limitée exclut expressément les dommages intentionnels.
Perturbed: Cette garantie limitée exclut expressément les dommages.
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


perturb_expansion_noimpact_es = """Task: You will be given a {{target_lang}} sentence and your goal is to perturb the sentence by adding one or two words in the sentence. Do not add words that change the intensity of the existing word. Please make sure that the added word does not disturb the grammaticality of the sentence and does not changes the meaning in a significant way. The added words would add more context that was already obvious from the sentence. Just output the perturbed {{target_lang}} sentence without giving any additional explanation.

*** Example Starts ***
Original: Cualquier controversia derivada de este contrato será resuelta por los tribunales competentes.
Perturbed: Cualquier controversia jurídica derivada de este contrato escrito será resuelta por los tribunales competentes.

Original: Los datos recopilados no se compartirán con terceros sin su consentimiento expreso.
Perturbed: Los datos personales recopilados no se compartirán con terceros ajenos sin su consentimiento expreso.
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


perturb_omission_es = """Task: You will be given a {{target_lang}} sentence and your goal is to perturb the sentence by removing information from the sentence. Remove only one or two words from the sentence. Please make sure that the removed information does not disturb the grammaticality of the sentence but should change the meaning in a significant way. Just output the perturbed {{target_lang}} sentence without giving any additional explanation.

*** Example Starts ***
Original: El arrendatario se compromete a pagar la renta y los gastos de comunidad.
Perturbed: El arrendatario se compromete a pagar la renta.

Original: La empresa garantiza la reparación o sustitución del producto defectuoso.
Perturbed: La empresa garantiza la reparación del producto defectuoso.
*** Example Ends ***

Original: {{original}}
Perturbed: """

prompts = {
    # French
    "synonym_fr": perturb_synonym_fr,
    "expansion_noimpact_fr": perturb_expansion_noimpact_fr,
    "alteration_fr": perturb_alteration_fr,
    "omission_fr": perturb_omission_fr,

    # Spanish
    "synonym_es": perturb_synonym_es,
    "expansion_noimpact_es": perturb_expansion_noimpact_es,
    "alteration_es": perturb_alteration_es,
    "omission_es": perturb_omission_es
}