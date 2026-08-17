"""Замороженный порядок 73 признаков классификатора.

Снят прогоном vendored-цепочки (insertion-order словарей PsyCues и PsyDict);
именно в этом порядке обучалась модель. Проверяется при каждом извлечении.
"""

CUES_NAMES = [
    'char_count', 'word_count', 'sentence_count', 'unique_words_count',
    'punctuation_count', 'punctuation_per_word', 'uppercase_rate',
    'mean_word_len', 'mean_sentence_len', 'unique_words_rate',
    'verbs_1p_rate', 'verbs_2p_rate', 'verbs_3p_rate',
    'verbs_past_tense_rate', 'infinitives_rate',
    'pro_1p_rate', 'pro_1p_sing_rate', 'pro_1p_plural_rate',
    'pro_2p_rate', 'pro_3p_rate',
    'trager_coef', 'logical_coh_coef', 'verbs_per_nouns_coef',
    'participles_gerunds_coef', 'negation_rate',
    'postag_A', 'postag_ADV', 'postag_ADVPRO', 'postag_ANUM', 'postag_APRO',
    'postag_COM', 'postag_CONJ', 'postag_INTJ', 'postag_NUM', 'postag_PART',
    'postag_PR', 'postag_S', 'postag_SPRO', 'postag_V',
]

PSY_DICT_NAMES = [
    'tgw_positive_assessment', 'tgw_positive_social', 'tgw_positive_emotions',
    'tgw_negative_assessment', 'tgw_negative_social', 'tgw_negative_emotions',
    'tgw_motivation_activity', 'tgw_cognitive_communication',
    'tgw_destructive_activity', 'tgw_affect_lex', 'tgw_bodily_states_emotions',
    'tgw_invectives', 'tgw_soft_invectives', 'tgw_obscene_lex',
    'tgw_youth_jargon', 'tgw_hcs', 'tgw_economics', 'tgw_catastrophes',
    'tgw_security_structures', 'tgw_healthcare_demography_ecology',
    'tgw_authority',
    'be_disgust', 'be_shame', 'be_anger', 'be_fear', 'be_sadness',
    'be_calm_excitement', 'be_happyness', 'be_wonder',
    'ew_positive', 'ew_negative', 'ew_ambivalent', 'ew_de_emotives',
    'sentiment_rate',
]

FEATURE_NAMES = CUES_NAMES + PSY_DICT_NAMES

assert len(CUES_NAMES) == 39 and len(PSY_DICT_NAMES) == 34 and len(FEATURE_NAMES) == 73
