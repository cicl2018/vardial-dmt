https://docs.google.com/document/d/1V5_MPpmbrsqeKUEMRyayJiuYpRdjSWe_PzEmHnMG3yI/edit


# 我想死
# 頭好痛
# 睡不够
# 日長睡起無情思，閒看兒童捉柳花。

| |data |
|---|:--|
|training:|first 16,770 sentences (simplified)|
|test:|last 2,000 sentences (simplified)|

# character-level

||unigrams|
|---|:--|
|TRAIN:|0.8899...|
|TEST:|80.25 %|
|finished:|16.774 seconds|

||bigrams|
|---|:--|
|TRAIN:|0.99946...|
|TEST:|84.75 %|
|finished:|177.633 seconds|

||trigrams|
|---|:--|
|TRAIN:|1.0|
|TEST:|87.75 %|
|finished:|740.108 seconds|

||4-grams|
|---|:--|
|TRAIN:|1.0|
|TEST:|86.5 %|
|finished:|1411.249 seconds|

# phrase-level
(collapsed whitespaces?)

||unigrams|
|---|:--|
|TRAIN:|0.999...|
|TEST:|84.8999 %|
|finished:|100.6239 seconds|

||bigrams|
|---|:--|
|TRAIN:|1.0|
|TEST:|78.6 %|
|finished:|420.181 seconds|

||trigrams|
|---|:--|
|TRAIN:|1.0|
|TEST:|67.65 %|
|finished:|506.467 seconds|

||4-grams|
|---|:--|
|TRAIN:|1.0|
|TEST:|57.45 %|
|finished:|471.756 seconds|
