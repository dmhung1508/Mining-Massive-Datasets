# Week 11 Demo Cases - Hung

## Muc tieu

- Chon san cac case de demo pipeline LSH va narrative clusters.
- Co vi du cluster lon, exact duplicate, va near duplicate.
- Dung cho task tuan 11: chuan bi demo cases va test demo.

## Top clusters

| cluster_id | cluster_size | sample_tweet_ids | sample_text |
| --- | --- | --- | --- |
| 1744 | 60 | 1566105622046982149, 1569845632227233793, 1571045843826003973 | #Russia Soldiers, cowards #Putin Don't obey! Throw away your weapons and go home! And eat a hot dinner with the whole family. Stay close to your family and l... \| #Russia Soldiers, cowards #Putin Don't obey! Throw away your weapons and go home! And eat a hot dinner with the whole family. Stay close to your family and l... \| #Russia Soldiers, cowards #Putin Don't obey! Throw away your weapons and go home! And eat a hot dinner with the whole family. Stay close to your family and l... |
| 26630 | 57 | 1633251318562189313, 1633251802157142017, 1633259718314475525 | This is going to be iconic. #Tbilisi #Georgia https://t.co/6zv7Md5mE2 \| This is going to be iconic. #Tbilisi #Georgia https://t.co/6zv7Md5mE2 \| This is going to be iconic. #Tbilisi #Georgia https://t.co/6zv7Md5mE2 |
| 35358 | 52 | 1643480275525967874, 1643511537909260289, 1643518687897419777 | Relax. It is not #Ukraine Just a holy place of worship in occupied Jerusalem. https://t.co/QWqwqPgKtj \| Relax. It is not #Ukraine Just a holy place of worship in occupied Jerusalem. https://t.co/QWqwqPgKtj \| Relax. It is not #Ukraine Just a holy place of worship in occupied Jerusalem. https://t.co/QWqwqPgKtj |
| 36360 | 50 | 1644863675591716864, 1644869103536971776, 1644870326336249856 | To @elonmusk 1. Is this a violation of TOS calling for genocide of #Ukraine 2. How is a terrorist state verified 3. Why did you allow #Russian leaders back o... \| To @elonmusk 1. Is this a violation of TOS calling for genocide of #Ukraine 2. How is a terrorist state verified 3. Why did you allow #Russian leaders back o... \| To @elonmusk 1. Is this a violation of TOS calling for genocide of #Ukraine 2. How is a terrorist state verified 3. Why did you allow #Russian leaders back o... |
| 31288 | 47 | 1638499744581296129, 1638499745483169795, 1638499808175443968 | A peaceful day in the world, another horrible day in Ukraine. Russian rocket just hit a section of residential buildings in Zaporizhzhia. Rescuers are workin... \| A peaceful day in the world, another horrible day in Ukraine. Russian rocket just hit a section of residential buildings in Zaporizhzhia. Rescuers are workin... \| A peaceful day in the world, another horrible day in Ukraine. Russian rocket just hit a section of residential buildings in Zaporizhzhia. Rescuers are workin... |

## Exact duplicate examples

| tweet_id_left | tweet_id_right | jaccard | left_text | right_text |
| --- | --- | --- | --- | --- |
| 1560520700485582848 | 1564805183632195585 | 1.0 | Ewch i fuck eich hun, Putin! (Welsh) | Ewch i fuck eich hun, Putin! (Welsh) |
| 1560520700485582848 | 1569970215538106368 | 1.0 | Ewch i fuck eich hun, Putin! (Welsh) | Ewch i fuck eich hun, Putin! (Welsh) |
| 1560520700485582848 | 1594526020102770690 | 1.0 | Ewch i fuck eich hun, Putin! (Welsh) | Ewch i fuck eich hun, Putin! (Welsh) |
| 1560520700485582848 | 1595440798471946240 | 1.0 | Ewch i fuck eich hun, Putin! (Welsh) | Ewch i fuck eich hun, Putin! (Welsh) |
| 1560520700485582848 | 1596077746198904832 | 1.0 | Ewch i fuck eich hun, Putin! (Welsh) | Ewch i fuck eich hun, Putin! (Welsh) |

## Near duplicate examples

| tweet_id_left | tweet_id_right | jaccard | left_text | right_text |
| --- | --- | --- | --- | --- |
| 1631430042587742208 | 1631471537202102272 | 0.978261 | I used to think that way until what I saw in my life was unforgettable &amp; unimaginable #TigrayGenocide has less voice, less media &amp; less attention but more damage to human lives &amp; infrastructures compared t... | I used to think that way until what I saw in my life was unforgettable &amp; unimaginable #TigrayGenocide has less voice, less media &amp; less attention but more damage to human lives &amp; infrastructures compared t... |
| 1637699730237997056 | 1637761018989273091 | 0.977778 | Hadas a rape survivor told DW "...He tried to take me to the bush, but I refused. He told me that he had a knife &amp; a handgun. Then he beat me with the stick." So members of @UN_HRC &amp; @UN 🚩Address #Justice4Tigr... | Hadas a rape survivor told DW "...He tried to take me to the bush, but I refused. He told me that he had a knife &amp; a handgun. Then he beat me with the stick." So members of @UN_HRC &amp; @UN 🚩Address #Justice4Tigr... |
| 1638071859928342529 | 1638128693745790980 | 0.977778 | @FitwiDesta The difference is #Ukrainian have blue eye ,#Tigrayan black eye ,in contrary tigray is the most who suffer more lost 1 millions lives by 🇪🇹 &amp; 🇪🇷 forces #Abiy &amp; #Isias must held accountable for the... | The difference is #Ukrainian have blue eye ,#Tigrayan black eye ,in contrary tigray is the most who suffer more lost 1 millions lives by🇪🇹 &amp; 🇪🇷 forces #Abiy &amp; #Isias must held accountable for the crimes they b... |
| 1562380177266671616 | 1562388943781371904 | 0.97619 | @UKRinPL Who #StandWithUkraine 1stU.S.of America and 2nd Poland CE EU* 3th GB &gt;&gt;DE FR 10xtimes less then! as % theim GDP (west EU planctonic donation arms)...meantime Putin'murderers DE'CORPO turnover rise up39%... | Who #StandWithUkraine 1stU.S.of America and 2nd Poland CE EU* 3th GB &gt;&gt;DE FR 10xtimes less then! as % theim GDP (west EU planctonic donation arms)...meantime Putin'murderers DE'CORPO turnover rise up39% Q1Q2 202... |
| 1637548093766860801 | 1637626741668626433 | 0.97619 | Putin is wanted by the ICC for war crimes if so #Isaias &amp; @AbiyAhmedAli must wanted by @IntlCrimCourt for the same crimes We all HUMAN We urge Equal access to Justice for All.#Justice4Tigray @UNHumanRights @UN @EU... | Putin is wanted by the ICC for war crimes if so , #Isaias &amp; @AbiyAhmedAli must wanted by @IntlCrimCourt for the same crimes . We all HUMAN‼️ We urge Equal access to Justice for All. #Justice4Tigray @UNHumanRights... |

## Demo script ngan

1. Mo bang top clusters de chi ra cum noi dung lap lon nhat.
2. Chon mot exact duplicate pair de giai thich Jaccard = 1.0.
3. Chon mot near duplicate pair de giai thich vi sao raw text khac nhung shingles van overlap cao.
4. Chay query demo voi mot cau trong exact duplicate de tra ve cac bai viet tuong tu.
5. Ket luan bang precision/recall va candidate reduction trong benchmark.
