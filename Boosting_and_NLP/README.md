# Boosting, Feature Generation and Natural Language Processing (NLP)

This directory take a stab at adding humor to NLP and Extreme Gradient Boosting. I take a dataset with excellent predictive features, and eliminate the 10 most important features. Below is an image of the feature importance scores from a baseline random forest classifier on the [Forest Cover Type Kaggle Dataset](https://www.kaggle.com/uciml/forest-cover-type-dataset). 

![](images/feature_importances.png?raw=true)

Feature importances, as the name suggests, rate how important each feature is for the decision a tree makes. To generate these scores, I used a [random forest classifier](Feature_Importances.ipynb). Feature importance scores are normalized between "0" and "1", where "0" indicates the feature is not used at all and "1" indicates the feature perfectly predicts the target. As can be seen above, 10 of the features are excellent predictors, whereas the other 44 are very poor predictors. Besides Feature 15 ('Wilderness_Area4': "Cache la Poudre Wilderness Area"), the other 43 features are terrible predictors. 

#### It seems like the logical thing to do would be to eliminate the first 10 (MOST IMPORTANT) features and only use the remaining 44 (LEAST IMPORTANT) features.  

-----

Clearly, this is not a wise decision. Nonetheless, I take on the challenge of using only these 44 features and see how well I can predict the 7 classes. 

For more information on the dataset, you can check out [previous EDA](https://github.com/adamszabunio/Forest_Cover_Type/tree/master/EDA) and [model fitting](https://github.com/adamszabunio/Forest_Cover_Type/tree/master/Random_Forests). Additonal information in the Appendix.


First, due to large class imbalances, I randomly undersample the dataset to have an equal number of samples for each class. 
- Note, this reduces the dataset to 3.3% of the original size. 
- ≈3% of the data and the 44 least informative features... This is gonna be fun!

Next, I [transform the worst 44 features into collocations](Feature_Reduction.ipynb). 
- This is a method of joining a sequence of words or terms which co-occur more often than would be expected by chance. Thank you [wikipedia](https://en.wikipedia.org/wiki/Collocation_extraction) for putting it succinctly. 
- An example from the data set would be to transform "Rawah Wilderness Area" --> "Rawah_Wilderness_Area", 3 words to 1. 
- In other words, I am creating new text "features" for NLP by transforming the category that the sample falls into. 

Last, I add new features by extracting additional information from map unit keys. With a little digging around, I was able to make sense of the soil type's ELU codes (4 digit code associated with each soil type). No pun intended.
- An example of an ELU code and associated description (with updated collocations):
- '2702': 'cathedral_family rock_outcrop_complex extremely_stony'
- From the ELU code:'2702', 
    - the first digit "2" corresponds to the climatic zone
        - "2" maps to the "lower montane" climatic zone
    - the second digit "7" corresponds to the geologic zones
        - "7" maps to the "ligneous and metamorphic" geologic zone

The final representation of the sample now looks like: 
- 'cathedral_family rock_outcrop_complex extremely_stony lower_montane, ligneous_and_metamorphic'

From 54 features ---> 5. Bring it on!

After transforming all samples in this manner, the dataset is ready for analysis. 

### NLP
--------
I begin with Topic Modeling. [NMF (Non-Negative Matrix Factorization) and LDA (LatentDirichletAllocation)](NMF_LDA.ipynb)

Wow, Topic Modeling was not the way to go. Even with my hack accuracy metric of ≈60% 'accuracy'

Instead, lets try two extremes:
- A classic NLP Classification model, Naive Bayes (as a baseline)
- XGBoost (Xtreme Gradient Boosting) 

XGBoost may be a bit of overkill in this case, but hey, this repo hasn't been about good decisions. Let's keep the trend going. [Naive Bayes and XGBoost](Naive_Boosting.ipynb)


Below is an image of the log loss score for the train and test set of a XGBoost model with 1000 estimators.

![](images/train_test_deviance.png?raw=true)

Even with XGBoost, it is apparent the model's learning plateaus around ≈300 estimators. 

### Conclusion
---------------

Garbage in. Garabge out. Silver lining.  With only 5 features per sample, and a vocabulary of 74 words, an f1 score of .60 for baseline models is not terrible. 


Appendix: 
--------
Context
-------
This dataset contains tree observations from four areas of the Roosevelt National Forest in Colorado. All observations are cartographic variables (no remote sensing) from 30 meter x 30 meter sections.

Classes to predict
------------
There are seven tree types, each represented by an integer variable:

1. Spruce/Fir 
2. Lodgepole Pine 
3. Ponderosa Pine 
4. Cottonwood/Willow 
5. Aspen 
6. Douglas-fir 
7. Krummholz

Most informative features:
------------------------
- Elevation: Elevation in meters 
- Aspect: Aspect in degrees azimuth 
- Slope: Slope in degrees 
- Horizontal Distance To Hydrology: Horz Dist to nearest surface water features 
- Vertical Distance To Hydrology: Vert Dist to nearest surface water features 
- Horizontal Distance To Roadways: Horz Dist to nearest roadway 
- Hillshade 9am (0 to 255 index): Hillshade index at 9am, summer solstice 
- Hillshade Noon (0 to 255 index): Hillshade index at noon, summer solstice 
- Hillshade 3pm (0 to 255 index): Hillshade index at 3pm, summer solstice 
- Horizontal Distance To Fire Points: Horz Dist to nearest wildfire ignition points 
- Wilderness Area (4 binary columns, 0 = absence or 1 = presence): Wilderness area designation 
- Soil Type (40 binary columns, 0 = absence or 1 = presence): Soil Type designation 
- Cover Type (7 types, integers 1 to 7): Forest Cover Type designation

Wilderness areas that are reduced to single collocations
------------------------
1. Rawah Wilderness Area 
2. Neota Wilderness Area 
3. Comanche Peak Wilderness Area 
4. Cache la Poudre Wilderness Area

Soil types also reduced to collocations (1-4, depending on soil type)
------------------

1. Cathedral family - Rock outcrop complex, extremely stony 
2. Vanet - Ratake families complex, very stony 
3. Haploborolis - Rock outcrop complex, rubbly 
4. Ratake family - Rock outcrop complex, rubbly 
5. Vanet family - Rock outcrop complex complex, rubbly 
6. Vanet - Wetmore families - Rock outcrop complex, stony 
7. Gothic family 
8. Supervisor - Limber families complex 
9. Troutville family, very stony 
10. Bullwark - Catamount families - Rock outcrop complex, rubbly 
11. Bullwark - Catamount families - Rock land complex, rubbly. 
12. Legault family - Rock land complex, stony 
13. Catamount family - Rock land - Bullwark family complex, rubbly 
14. Pachic Argiborolis - Aquolis complex 
15. unspecified in the USFS Soil and ELU Survey 
16. Cryaquolis - Cryoborolis complex 
17. Gateview family - Cryaquolis complex 
18. Rogert family, very stony 
19. Typic Cryaquolis - Borohemists complex 
20. Typic Cryaquepts - Typic Cryaquolls complex 
21. Typic Cryaquolls - Leighcan family, till substratum complex 
22. Leighcan family, till substratum, extremely bouldery 
23. Leighcan family, till substratum - Typic Cryaquolls complex 
24. Leighcan family, extremely stony 
25. Leighcan family, warm, extremely stony 
26. Granile - Catamount families complex, very stony 
27. Leighcan family, warm - Rock outcrop complex, extremely stony 
28. Leighcan family - Rock outcrop complex, extremely stony 
29. Como - Legault families complex, extremely stony 
30. Como family - Rock land - Legault family complex, extremely stony 
31. Leighcan - Catamount families complex, extremely stony 
32. Catamount family - Rock outcrop - Leighcan family complex, extremely stony 
33. Leighcan - Catamount families - Rock outcrop complex, extremely stony 
34. Cryorthents - Rock land complex, extremely stony 
35. Cryumbrepts - Rock outcrop - Cryaquepts complex 
36. Bross family - Rock land - Cryumbrepts complex, extremely stony 
37. Rock outcrop - Cryumbrepts - Cryorthents complex, extremely stony 
38. Leighcan - Moran families - Cryaquolls complex, extremely stony 
39. Moran family - Cryorthents - Leighcan family complex, extremely stony 
40. Moran family - Cryorthents - Rock land complex, extremely stony


