# Academic Evaluation Report

**Model:** `/lambda/nfs/lambda-cloud-data/sumerian_translation/models/sumerian_mt5_final`

**Test Set:** `/lambda/nfs/lambda-cloud-data/sumerian_translation/output_training_v2_clean/finetune/valid.jsonl` (661 examples)

## 1. Quantitative Metrics

| Metric | Score |
|--------|-------|
| BLEU | 0.14 |
| chrF++ | 7.28 |

## 2. Score Distribution

- **Mean prediction length:** 37.6 words
- **Mean reference length:** 42.8 words

## 3. Qualitative Error Analysis

Lowest-scoring predictions for manual review:

### Example 106

**Sumerian:** `lugal-mar2-da-ke4 iri-ni-ta bar-ta ba-da-gub nin-zu-an-na ki-tuc ki aĝ2-ĝa2-ni ĝiri3 kur2 ba-ra-an-dab5 a iri gul-la e2 gul-la-ĝu10 gig-ga-bi im-me i3-si-in ec3 kar-re nu-me-a a-e ba-e-dar nin-isin2-na ama kalam-ma-ke4 er2 gig mu-un-ce8-ce8 a iri gul-la e2 gul-la-ĝu10 gig-ga-bi im-me en-lil2-le dur-an-ki-ka mitum2-a ba-an-sag3 en-lil2-le iri-ni ec3 nibru-a a-nir ba-ab-ĝar ama nin-lil2 nin ki-ur3-ra-ke4 er2 gig mu-un-ce8-ce8 a iri gul-la e2 gul-la-ĝu10 gig-ga-bi im-me`

**Reference:** Lugal-Marda stepped outside his city. Ninzuana took an unfamiliar path away from her beloved dwelling. Alas, the destroyed city, my destroyed house, she cried bitterly. Isin, the shrine that was not a quay, was split by onrushing waters. Ninisina, the mother of the Land, wept bitter tears. Alas, the destroyed city, my destroyed house, she cried bitterly. Enlil smote Dur-an-ki with a mace. Enlil made lamentation in his city, the shrine Nibru. Mother Ninlil, the lady of the Ki-ur shrine, wept bitter tears. Alas, the destroyed city, my destroyed house, she cried bitterly.

**Prediction:** The king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king

**Similarity:** 0.0000

---

### Example 394

**Sumerian:** `a-ba-am3 sahar mu-un-de6 mu-un-zig3 a-ba-am3 ma2 bi2-in-du8 a-ba-am3 cir3-e dur2-dur2-ru-da mu-un-…`

**Reference:** = Alster 1997 6.50; cf. 6.1.07.78 Who moved 1 ms. has instead: removed the dust? Who caulked the boat? Who …… while they sat singing?

**Prediction:** In the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the 

**Similarity:** 0.0000

---

### Example 393

**Sumerian:** `pisaĝ-ninda-ĝar en-lil2-la2-ka i3 ku6 i3 mucen-na-ka gu2-zu-ce3 SAR-re-ec2 he2-em-DU`

**Reference:** = Alster 1997 6.49 May …… fish oil and bird oil on your shoulders for the offering basket of Enlil.

**Prediction:** The king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king

**Similarity:** 0.0000

---

### Example 291

**Sumerian:** `buru14-da gu7-e lag nu-bur12-re cu i3-bur12-ra numun nu-ĝa2-ĝa2`

**Reference:** cf. 6.1.26.d5 He who eats during the harvest is not removing clods. He who tears out weeds (?) is not sowing seed.

**Prediction:** The king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king

**Similarity:** 0.0000

---

### Example 490

**Sumerian:** `ninda ĝectin-da ma-da-du-du-nam ĝen nin9 ki aĝ2-ĝu10 cag4-bi ga-am3-mi-ib-X na-na-a ne ge4-su-ub`

**Reference:** You come to me with bread and wine. Come, my beloved sister, let me …… this heart. Nanaya, let me kiss you.

**Prediction:** The king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king

**Similarity:** 0.0000

---

### Example 272

**Sumerian:** `ka5-a-a dam-a-ni an-na-ab-be2 ĝa2-nu unug ga-rac-gin7 zu2 ga-am3-gaz-e-en-de3-en kul-aba e-sir2-gin7 ĝiri3-me-a ga-am3-ma-ab-sig9-ge4-en-de3-en iri-ce3 600 uc nu-te-a-ba iri-da ur-re ceg11 am3-da-gi4-gi4 geme2-tum-ma-al geme2-tum-ma-al dur2-zu-ce3 ĝa2-nam-ma-da iri-da niĝ2-hul-e ceg11 am3-da-gi4-gi4`

**Reference:** The fox said to his wife: Come! Let us crush Unug between our teeth like a leek; let us strap Kulaba on our feet like sandals! Before they had yet come within a distance of 600 uš from the city, the dogs began to howl from the city. -- Geme-Tummal! Geme-Tummal! Come with me to your place! Wicked things are howling at us from the city!

**Prediction:** In the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the 

**Similarity:** 0.0000

---

### Example 274

**Sumerian:** `lul dug4-ga-ab zid dug4-ga-ab lul ba-e-sig10-ge5`

**Reference:** cf. 6.1.07.89 Tell a lie and then tell the truth: it will be considered a lie.

**Prediction:** In the mountains of the mountains of the mountains, the mountains of the mountains, the mountains of the mountains, the mountains of the mountains, the mountains of the mountains, the mountains of the mountains, the mountains of the mountains, the mountains of the mountains, the mountains of the mountains, the mountains of the mountains, the mountains of the mountains, the mountains of the mountains, the mountains of the mountains, the mountains of the mountains, the mountains of the mountains, the 

**Similarity:** 0.0000

---

### Example 485

**Sumerian:** `… X … X … X … X … X`

**Reference:** 

**Prediction:** The ...... ...... ...... ...... .......

**Similarity:** 0.0000

---

### Example 55

**Sumerian:** `iri u3-mu-niĝin2 bad3 gen6 um-mi-du3 e2 diĝir gal-gal-e-ne-ka pa e3 u3-ba-ni-ak im su4 im sig7 im dal-ha-mun-na mi2 um-ma-ni-in-dug4 iri e2-gal-la-ka i3-du3-e-en ugula nu-banda3-e-ne dur2 i-im-ĝa2-ĝa2-ne`

**Reference:** When I have gone through the city and built its sturdy walls, have made the temples of the great gods splendid and embellished them with brown, yellow and decorative (?) clay, I build in the city of the palace where the inspectors and overseers live.

**Prediction:** The king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king

**Similarity:** 0.0000

---

### Example 279

**Sumerian:** `ance gu3 an-mur lugal ance-ke4 pa-aĝ2 an-ze2 ba-da-ra-ab-ed3-de3-en-de3-en kac4-a ĝen-na-e-ce`

**Reference:** The donkey roared (?); its owner pierced its nostrils (?): We must get up and away from here! Quickly! Come!

**Prediction:** In the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the 

**Similarity:** 0.0000

---

### Example 131

**Sumerian:** `a e2 zid e2 zid a lu2-bi lu2-bi`

**Reference:** O good house, good house! O its people, its people!

**Prediction:** In the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the 

**Similarity:** 0.0000

---

### Example 569

**Sumerian:** `X … A …`

**Reference:** 

**Prediction:** The ...... ...... .......

**Similarity:** 0.0000

---

### Example 73

**Sumerian:** `ud re-a nam ba-tar-ra-ba mu he2-ĝal2 an u3-tud-da uĝ3-e u2-cim-gin7 ki in-dar-ra-ba en abzu lugal en-ki-ke4 en-ki en nam tar-tar-re-de3 e2-a-ni kug za-gin3-na tec2-bi ba-ni-in-du3 kug za-gin3-bi ud kar2-kar2-a-ka ec3-e abzu-a ul im-ma-ni-in-de6`

**Reference:** In those remote days, when the fates were determined; in a year when An brought about abundance, and people broke through the earth like green plants -- then the lord of the abzu, King Enki, Enki, the lord who determines the fates, built up his temple entirely from silver and lapis lazuli. Its silver and lapis lazuli were the shining daylight. Into the shrine of the abzu he brought joy.

**Prediction:** The king of the king of the king of the king of the king of the king of the king of the king of the king of the king of the king of the king of the king of the king of the king of the king of the king of the king of the king of the king of the king of the king of the king of the king of the king of the king of the king of the king of the king of the king of the king of the king

**Similarity:** 0.0000

---

### Example 67

**Sumerian:** `cubur-e cu mu2-mu2 gal-an-zu X X AN cubur-e cu mu2-mu2 gal-an-zu MU X akkil-ke4 cu mu2-mu2 gal-an-zu iri-na akkil-a dur2 ba-ni-in-ĝar cubur-e akkil-a dur2 ki ba-ni-in-ĝar cubur-e X cu AN SI X Xbr ga-ca-an-cubur X nin-tur5 cu AN SI X Xbr cubur cu AN SI … ga-ca-an-cubur … an-ra X X X mu-e-ni-il2-X`

**Reference:** The servant (šubur), the wise suppliant, …… the servant, the wise suppliant, the …… of Akkil, the wise suppliant has taken her seat in her city Akkil. The servant has taken her seat in Akkil. The servant ……, Ninšubur, …… Nintur, the servant ……, Ninšubur …… to An.

**Prediction:** In the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the 

**Similarity:** 0.0000

---

### Example 62

**Sumerian:** `me-en-de3 en-te-en buru14-gin7 mu-e-la2 cu e2-me-ec en-te-na-ke4 mu-na-kar-kar-re-en-de3-en al-e a2 la2-e garadinx mu-un-la2 har-mucen-na a2 la2-e gi-gur-ra mu-un-la2 ĝuruc saĝ-dili ki-gul-la a2 mu-un-da-an-e3 an pad-pax-ra2 im-de5-de5-ge-ne`

**Reference:** For us you raise winter like the harvest-time. We take away the hand of summer and winter. Hoe, the binder, ties the sheaves. Binding bird-traps, it ties the reed-baskets. The solitary labourer and the destitute are supported. 2 mss. add 1 line: They glean the scattered ears.

**Prediction:** In the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the 

**Similarity:** 0.0000

---

### Example 60

**Sumerian:** `apin sur3 ba-al-ba-al in-ce3 mu-e-dub2 edin bar-rim4 ki a nu-ĝal2-la a dug3-ga-bi u3-mu-ba-al lu2 enmen tuku gu2 pu2-ĝa2-ce3 zi-ni ba-ci-in-tum3`

**Reference:** Insultingly you call me "Plough, the digger of ditches". But when I have dug out the fresh water for the plain and dry land where no water is, those who have thirst refresh themselves at my well-head.

**Prediction:** The king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king

**Similarity:** 0.0000

---

### Example 579

**Sumerian:** `a-ru-ru nin9 mu-ul-lil2-la2-ke4 iri-ni iri-saĝ-rig7 zag he2-bi2-in-tag kec3 ki-ulutim2 kalam-ma-ke4 itima kug ud nu-zu-ba uĝ3-e igi he2-ni-in-bar zag he2-bi2-in-tag cu li-bi2-in-tag kiĝ2-sig unu2 gal-ni mu-ni he2-pad3-de3 u3-mu-un nanna u3-mu-un ac-im2-babbar iri-ni urim2-ma zag he2-bi2-in-tag cag4-mar-ra-ka ka-na-aĝ2-bi he2-en-til e2-kic-nu-ĝal2 aĝ2-gig-ga he2-bi2-ak cag4-bi he2-bi2-ra zag he2-bi2-in-tag cu li-bi2-in-dag kiĝ2-sig unu2 gal-ni mu-ni he2-pad3-de3`

**Reference:** Aruru, the sister of Enlil, destroyed her city Iri-saĝ-rig. In Keš, the creation place of the Land, the people saw inside its holy sanctuary where daylight had been unknown. She destroyed it but did not abandon it -- at the lunches, in her great dining hall, they call her name. Lord Nanna, Lord Ašimbabbar, destroyed his city Urim. He decimated the Land with famine. He committed a sacrilege against the E-kiš-nu-ĝal. He struck at its heart. He destroyed it but did not abandon it -- at the lunches, in his great dining hall, they call his name.

**Prediction:** The king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king

**Similarity:** 0.0000

---

### Example 585

**Sumerian:** `u3-mu-un am-an-ki uru2-zu uru2-ze2-eb gul-la a-ba-a mu-ni-in-du8 ec3 abzu e2-zu-gin7 hul-a a-ba-a igi mu-ni-in-du8 ki-cu-tag-ga-ni lu2 nu-ed3-de3 kiĝ2-sig unu2 gal-ni mu-ni li-bi2-in-pad3-de3 en-ki lugal abzu-ke4 cag4 ba-an-sag3 ur5-ra-ni ba-uc2 inim nitalam-na-ce3 ni2-te-a-ni i-si-ic mi-ni-ib-la2 cag4-ka-tab-ba ba-an-nu2`

**Reference:** Lord Enki, who has ever seen such a destruction as that of your city Eridug? Who has ever seen such a misfortune as that of the shrine Abzu, your house? No one goes up to his offering terrace. At the lunches, in his great dining hall, they do not call his name. Enki, king of the abzu, felt distressed, felt anxious. At the words of his spouse, he himself began to wail. He lay down and fasted.

**Prediction:** In the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the mountains, the 

**Similarity:** 0.0000

---

### Example 155

**Sumerian:** `i-lu cag4-ga mu bal sag9-ga nun-e nun uru16 dirig gal mah-bi ad-da-zu ic-me-da-gan lugal kalam-ma-ke4 gu-za-na suhuc-bi mu-ra-an-ge-en inim dug4-ga an en-lil2-la2-ta du17 nim kur-kur-ra si-a mu-ni-in-ĝar mu-e-ni-ĝar`

**Reference:** Among joyful songs of the heart, in an auspicious regnal year, the prince, the powerful prince surpassing in greatness and majesty, your father Išme-Dagan, king of the Land, made the foundations of his throne firm for you. On the orders of An and Enlil, he 1 ms. has instead: you silenced the loud (?) strife of the foreign countries.

**Prediction:** The king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king

**Similarity:** 0.0000

---

### Example 151

**Sumerian:** `li-pi2-it-ec4-tar2 lugal saĝ il2 nun barag-ga he2-du7 isimu2 nam-lugal-la utu-gin7 du ce-er-zid kalam-ma nam-nun-ce3 mah me gal-la u5 ub-da 4 uĝ3 ki ĝar-ra ce-ga en-lil2-la2 nin-lil2-le ki aĝ2 cul zid igi gun3 barag-ga tum2-ma men aga zid saĝ me-te-ĝal2 cibir cu du8 du7 saĝ gig2-ga nun li-pi2-it-ec4-tar2 dumu en-lil2-la2 sipad igi-ĝal2-tuku uĝ3 lah5-lah5-e ĝissu dug3-ga ud IC-e ni2 dub2-bu en alim mah an-ne2 ki aĝ2 ĝickim-til3-zu-um ama nin-lil2-la2 li-pi2-it-ec4-tar2 a2 nun hu-mu-te-ĝal2`

**Reference:** Lipit-Eštar, proud king, enthroned prince, most seemly offshoot of kingship, who walks like Utu, brilliant light of the Land, lofty in nobility, riding on the great divine powers; who settles the people in the four quarters; favoured by Enlil, beloved by Ninlil, trustworthy youth with shining eyes, worthy of the throne-dais, whose seemly head is adorned with the tiara, the good headdress, who holds in his hand 1 ms. has instead: is perfect with the sceptre over the black-headed, prince Lipit-Eštar, son of Enlil, wise shepherd, who leads the people to let them relax …… in pleasant shade, lord, great bison, beloved by An! Your trust is put in Mother Ninlil; Lipit-Eštar, you exert great power.

**Prediction:** The king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king, the king of the king

**Similarity:** 0.0000

---

## 4. Sample Good Predictions

- **Sumerian:** `ukur3 si-ig kalam-ma-ka...`
  - **Pred:** The king of the king of the king.
  - **Ref:** The poor are the weak in the Land.

- **Sumerian:** `sag9-ga-bi pad3-da hul-bi u2-gu de2-am3...`
  - **Pred:** The king of the king of the king of the king of the king.
  - **Ref:** The good thing is to find it; the bad thing is to lose it.

- **Sumerian:** `gud-de3 al-ur11-ru ur-re sur3 an-tag...`
  - **Pred:** The king of the king of the king of the king of the king.
  - **Ref:** While the ox is ploughing, the dog is spoiling the deep furrows.

- **Sumerian:** `id2 niĝ2 ĝal2-la he2-gid2-i...`
  - **Pred:** The king of the king of the king of the king.
  - **Ref:** Let the river expand when there is something in it.

- **Sumerian:** `iri lil2-la2-am3 cag4-bi a-ce-ra gi er2-ra ba-an-m...`
  - **Pred:** The king of the king of the king of the king of the king of the king of the king.
  - **Ref:** There is lamentation in the haunted city, mourning reeds grew there. In its midst there is lamentation, mourning reeds grew there. Its people spend their days in moaning.

- **Sumerian:** `nam-sag9-ga kac-am3 nam-hul kaskal-am3...`
  - **Pred:** The king of the king of the king of the king of the king.
  - **Ref:** cf. 6.1.07.98 The good thing is the beer. The bad thing is the journey.

- **Sumerian:** `ur cub6-ba saĝ KA.DU ha-ha-za me-ce3 gi4-mu-un-ze2...`
  - **Pred:** The king of the king of the king of the king of the king.
  - **Ref:** Slavering dogs waiting for instructions (?) ……: Where are you going? Come back! Stay!

- **Sumerian:** `ur ki-tuc-bi nu-mu-zu-a...`
  - **Pred:** The king of the king of the king of the king of the king.
  - **Ref:** A dog which knows no home.

- **Sumerian:** `gala-e gana2 e2-e us2-sa...`
  - **Pred:** The king of the king of the king of the king of the king.
  - **Ref:** To the lamentation priest the field lies adjacent to the house.

- **Sumerian:** `ki-ta kur2 niĝ2-gig nin-urta-ke4...`
  - **Pred:** The king of the king of the king of the king.
  - **Ref:** To remove something from its proper place is an abomination to Ninurta.

